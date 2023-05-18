#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 22:13
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : interactive_EVE.py
# @Describe:
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


from inference.data.mask_mapper import MaskMapper
from inference.inference_core_sam import InferenceCore
from util.util_sam.demo_utils import \
    Logger, read_logs, get_meta_from_video,  select_sorted, resize_mask
from util.plot_save import fvis

def init_EVE(vid_length=20):  # TODO:想办法每次都输入vid_length
    from model.model_sam.network_sam import EVE
    from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    from segment_anything import SamPredictor
    config = edict({
        'model': 'saves/vit_h_s2/vit_h_s2_70000.pth',
        'd16_path': '/DAVIS-2017-trainval-480p',
        'd17_path': '/DAVIS-2017-trainval-480p',
        'y18_path': '/YouTube2018',
        'y19_path': '/YouTube',
        'lv_path': '/long_video_set',
        'generic_path': None,
        'dataset': 'D17',
        'split': 'val',
        'output': 'output/D17_val_EVE',
        'save_all': False,
        'eval_on_dv17': True,
        'benchmark': False,
        'disable_long_term': False,
        'max_mid_term_frames': 10,
        'min_mid_term_frames': 5,
        'max_long_term_elements': 10000,
        'num_prototypes': 128,
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'model_type': 'vit_h',
        'embed_dim': 768,
        'save_scores': False,
        'flip': False,
        'size': 480,
        'enable_long_term': True})

    config['enable_long_term_count_usage'] = (
            config['enable_long_term'] and
            (vid_length
             / (config['max_mid_term_frames'] - config['min_mid_term_frames'])
             * config['num_prototypes'])
            >= config['max_long_term_elements']
    )
    network = EVE(config, config.model).cuda().eval()
    model_weights = torch.load(config.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)

    mapper = MaskMapper()
    auto_masker = SamAutomaticMaskGenerator(network.sam)
    prompt_pridictor = SamPredictor(network.sam)
    processor = InferenceCore(network, config=config)
    print("Initialize successfully.")
    return network, config, mapper, processor, auto_masker, prompt_pridictor

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


"""only used in Segment Anything"""
MAX_NUM_OBJ = 20
ALLOW_MIN_AREA = 100
torch.autograd.set_grad_enabled(False)
network, config, mapper, processor, auto_masker, prompt_predictor = init_EVE()

from PIL import Image
import os

PALETTE = Image.open("sam_scripts/assets/blackswan_00000.png").getpalette()

# 读取视频
blackswan_video = "sam_scripts/assets/cars.mp4"
blackswan_video = "sam_scripts/assets/cell.mp4"
blackswan_video = "sam_scripts/assets/blackswan.mp4"


def read_video(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.tensor(frame))
        else:
            break
    cap.release()
    frames = torch.stack(frames, dim=0).permute(0, 3, 1, 2)
    return frames


frames = read_video(blackswan_video)
from util.eval_util import get_preprocess_shape, preprocess, postprocess_masks

# image = cv2.imread("sam_scripts/assets/blackswan_00000.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = frames[0].permute(1, 2, 0).numpy()

"""为当前帧生成Segment Anything."""
# masks = auto_masker.generate(image)
# masks = select_sorted(masks, ALLOW_MIN_AREA, MAX_NUM_OBJ)


prompt_predictor.set_image(image)

"""使用纯BBoxes引导，每次可以选择多个目标xyxy"""
input_boxes = torch.tensor([
    [150, 65, 520, 380],
    [0, 0, 854, 200],
], device=prompt_predictor.device)

# Transform the boxes to the input frame, then predict masks. `SamPredictor` stores the necessary transform as the `transform` field for easy access, though it can also be instantiated directly for use in e.g. a dataloader (see `segment_anything.utils.transforms`).
transformed_boxes = prompt_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = prompt_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)  # (num_input) x (num_predicted_masks_per_input) x H x W
masks = masks.to(torch.uint8)

# """使用Click和initial_box引导,每次只能选择一个目标"""
# input_box = np.array([150, 65, 520, 380])
# input_point = np.array([[300, 300], [100, 100]])
# input_label = np.array([1, 0])  # neg:0, pos:1
#
# masks, _, _ = prompt_predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     box=input_box,
#     multimask_output=False,
# )
# masks = masks[None].astype(np.uint8)


"""curr_overlay检查"""
from util.util_sam.demo_utils import generate_masks_tensor

masks_tensor = generate_masks_tensor(masks)  # [num_obj+bg, h, w]
masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
# check_overlay(masks_np, image, PALETTE)

"""调整参考帧信息"""
msk, labels = mapper.convert_mask(masks_np)
msk = torch.Tensor(msk).cuda()
msk_device = msk.device
need_resize = True
if need_resize:
    msk = resize_mask(msk[None])[0]
processor.set_all_labels(list(mapper.remappings.values()))

import torchvision.transforms.functional as TF
from tqdm import tqdm

PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53], device=msk_device).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375], device=msk_device).view(3, 1, 1)

DEMO_OUTPUTS = "sam_scripts/demo_outputs"
vid_length = len(frames)

"""视频推理"""
for ti, rgb in tqdm(enumerate(frames)):
    rgb = frames[ti].to(msk_device)
    msk = msk if ti == 0 else None
    labels = labels if ti == 0 else None
    input_size = rgb.shape[-2:]
    target_size = get_preprocess_shape(*input_size, 1024)
    rgb = (rgb - PIXEL_MEAN) / PIXEL_STD
    rgb = preprocess(TF.resize(rgb, target_size), 1024)
    if msk is not None:
        msk = preprocess(TF.resize(msk, target_size), 1024)
    prob = processor.step(rgb, msk, labels, end=(ti == vid_length - 1))
    prob = postprocess_masks(prob, target_size, input_size)
    if need_resize:
        prob = F.interpolate(prob.unsqueeze(1), input_size, mode='bilinear', align_corners=False)[:, 0]
    out_mask = torch.argmax(prob, dim=0)
    out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
    this_out_path = os.path.join(DEMO_OUTPUTS)
    os.makedirs(this_out_path, exist_ok=True)
    out_mask = mapper.remap_index_mask(out_mask)
    out_img = Image.fromarray(out_mask)
    out_img.putpalette(PALETTE)
    out_img.save(os.path.join(this_out_path, f'{str(ti).zfill(5)}.png'))
