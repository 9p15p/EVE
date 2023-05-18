"""
This file defines EVE, the highest level nn.Module interface
During training, it is used by trainer_sam.py
During evaluation, it is used by inference_core_sam.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F
from model.aggregate import aggregate
from model.model_sam.build_sam import sam_model_registry
from model.modules import ValueEncoder, KeyProjection
from model.memory_util import *
from model.model_sam.common_sam import LayerNorm2d
from model.model_sam.modules_sam import HiddenUpdater
from segment_anything.utils.transforms import ResizeLongestSide
from util.plot_save import fvis


class EVE(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)
        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')

        sam = sam_model_registry[config.model_type](checkpoint=config.get('sam_checkpoint', None))
        # sam.to(device=device)
        self.sam = sam
        self.freeze_sam_params()

        self.key_encoder = self.sam.image_encoder
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object, 256)  # R18
        # Projection from f16 feature space to key/value space
        self.key_proj = KeyProjection(256, self.key_dim)
        self.prompt_encoder = self.sam.prompt_encoder
        self.prompt_encoder.batch_size = config.get('batch_size', 1)
        # self.decoder = Decoder(self.value_dim, self.hidden_dim)
        self.decoder = self.sam.mask_decoder

        if config.model_type in ['default', 'vit_h']:
            self.embed_dim = 1280
        elif config.model_type in ['vit_l']:
            self.embed_dim = 1024
        elif config.model_type in ['vit_b']:
            self.embed_dim = 768

        self.neck = nn.ModuleDict({
            'f8': self.build_neck(self.embed_dim, 128, 'f8'),
            'f4': self.build_neck(self.embed_dim, 64, 'f4'),
        })

        msk_embed_dim = self.sam.prompt_encoder.embed_dim
        mask_in_chans = msk_embed_dim + self.value_dim
        self.fuse_mem_dense_embeddings = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 4, 3, 1, 1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans // 8, 1, 1, 0),
            LayerNorm2d(mask_in_chans // 8),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 8, msk_embed_dim, 1, 1, 0),
        )
        self.hidden_update = HiddenUpdater([256 + 1, 128, 64], 256, hidden_dim=self.hidden_dim)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.original_size = (480, 910)
        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def fuse_mem_embed(self, memory_readout, dense_embeddings):
        """
        1.dense_embeddings先expand到和memory_readout一样的shape
        2.将两者concat到一起
        3.通过一个卷积层融合
        """
        bz, n, _, h, w = memory_readout.shape
        memory_readout = memory_readout.flatten(0, 1)
        dense_embeddings = torch.repeat_interleave(dense_embeddings, n, dim=0)
        tmp = torch.cat([memory_readout, dense_embeddings], dim=1)
        feat_fuse = self.fuse_mem_dense_embeddings(tmp)
        feat_fuse = feat_fuse.view(bz, n, -1, h, w)
        return feat_fuse

    def build_neck(self, embed_dim, out_chans, scale):
        assert scale in ['f4', 'f8']
        if scale == 'f4':
            neck = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                LayerNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
                nn.Conv2d(embed_dim // 4, out_chans, kernel_size=1, bias=False, ),
                LayerNorm2d(out_chans),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False, ),
                LayerNorm2d(out_chans),
            )
        elif scale == 'f8':
            neck = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                nn.Conv2d(embed_dim // 2, out_chans, kernel_size=1, bias=False, ),
                LayerNorm2d(out_chans),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False, ),
                LayerNorm2d(out_chans),
            )
        return neck

    def encode_key(self, frame, need_sk=True, need_ek=True):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        self.input_size = tuple(frame.shape[-2:])
        f16_origin, f16 = self.key_encoder(frame)
        f16_origin = f16_origin.permute(0, 3, 1, 2)
        f8 = self.neck['f8'](f16_origin)
        f4 = self.neck['f4'](f16_origin)
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True):
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i != j]],
                    dim=1,
                    keepdim=True)
                for i in range(num_objects)],
                dim=1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g16, h16

    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key,
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def preprocess_input_sam(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            boxes: Optional[np.ndarray] = None,
            mask_input: Optional[np.ndarray] = None,
    ):
        """Transform input prompts"""
        coords_torch, labels_torch, boxes_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                    point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if boxes is not None:
            boxes = self.transform.apply_boxes(boxes, self.original_size)
            boxes_torch = torch.as_tensor(boxes, dtype=torch.float, device=self.device)
            boxes_torch = boxes_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        return coords_torch, labels_torch, boxes_torch, mask_input_torch

    def cal_embeddings(self, point_coords, point_labels, boxes, mask_input):
        tmp = self.preprocess_input_sam(point_coords, point_labels, boxes, mask_input)
        point_coords, point_labels, boxes, mask_input = tmp
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        return sparse_embeddings, dense_embeddings

    def segment(self, multi_scale_features, memory_readout,
                hidden_state, selector=None, h_out=True, strip_bg=True,
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                boxes: Optional[np.ndarray] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = False,
                return_logits: bool = False,
                ):
        # point_coords = np.array([[520, 275]], float)
        # point_labels = np.array([1])
        # self.device = multi_scale_features[0].device

        # current prompts 没有意义，因为除了参考帧，其他帧不会有这四个输入
        sparse_embeddings, dense_embeddings = self.cal_embeddings(point_coords, point_labels, boxes, mask_input)
        # historical prompt
        if mask_input is None:
            dense_embeddings_fused = self.fuse_mem_embed(memory_readout, dense_embeddings)
        else:  # 使用当前交互mask进行修正
            raise NotImplementedError
        """
        1.将multi_scale_features[0]附带带有历史特征的dense_embeddings: 即src
        2.结合预测结果low_res_logits, 更新hidden_state
        """

        # bz, n, _, h, w = memory_readout.shape
        # dense_embeddings = torch.repeat_interleave(dense_embeddings[None], n, dim=1)
        # low_res_logits, iou_predictions, g16 = self.decoder(
        #     image_embeddings=multi_scale_features[0],
        #     image_pe=self.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=multimask_output,
        # )

        # Predict masks
        low_res_logits, iou_predictions, g16 = self.decoder(
            image_embeddings=multi_scale_features[0],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings_fused,
            multimask_output=multimask_output,
        )
        low_res_logits = low_res_logits[:, :, 0]
        lower_res_logits = F.interpolate(low_res_logits, scale_factor=0.25, mode="bilinear", align_corners=False)
        b, n, h, w = lower_res_logits.shape
        lower_res_logits = lower_res_logits[:, :, None]
        g16 = g16.view(b, n, -1, h, w)
        g16 = torch.cat([g16, lower_res_logits], dim=2)
        g8 = torch.repeat_interleave(multi_scale_features[1].unsqueeze(1), repeats=g16.size(1), dim=1)
        g4 = torch.repeat_interleave(multi_scale_features[2].unsqueeze(1), repeats=g16.size(1), dim=1)
        hidden_state = self.hidden_update([g16, g8, g4], hidden_state)

        low_res_probs = torch.sigmoid(low_res_logits)
        if selector is not None:
            low_res_probs = low_res_probs * selector
        low_res_logits, low_res_probs = aggregate(low_res_probs, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            low_res_probs = low_res_probs[:, 1:]
        logits = F.interpolate(low_res_logits, scale_factor=4, mode="bilinear", align_corners=False)
        probs = F.interpolate(low_res_probs, scale_factor=4, mode="bilinear", align_corners=False)
        # prob=softmax(logits)[:, 1:]
        return hidden_state, logits, probs

        # # Upscale the masks to the original image resolution
        # # TODO: input_size运行时的大小，original_size是原始大小
        # self.original_size = self.input_size
        # b,n = low_res_logits.shape[:2]
        # masks = self.sam.postprocess_masks(low_res_logits.flatten(0,1), self.input_size, self.original_size)
        # masks = masks.view(b,n,*masks.shape[-2:])
        # print("hello")

        # hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        # prob = torch.sigmoid(logits)
        # if selector is not None:
        #     prob = prob * selector
        #
        # logits, prob = aggregate(prob, dim=1, return_logits=True)
        # if strip_bg:
        #     # Strip away the background
        #     prob = prob[:, 1:]
        # # prob=softmax(logits)[:, 1:]
        # return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['hidden_update.transform.weight'].shape[0] // 3
            print(f'Hyperparameters read from the model weights: '
                  f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict)

    def freeze_sam_params(self):
        for param in self.sam.parameters():
            param.requires_grad = False
