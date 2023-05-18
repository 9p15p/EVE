#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 13:30
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : demo_utils.py
# @Describe:
import sys
import os
import gradio as gr
import cv2
import numpy as np
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import zipfile
import shutil

from inference.data.mask_mapper import MaskMapper
from inference.inference_core_sam import InferenceCore
from util.eval_util import get_preprocess_shape, preprocess, postprocess_masks
from util.plot_save import fvis

torch.autograd.set_grad_enabled(False)
PALETTE = Image.open("sam_scripts/assets/blackswan_00000.png").getpalette()


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        if os.path.exists(filename):
            os.remove(filename)
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()


def get_meta_from_video(input_video):
    if input_video is None:
        raise gr.Error("input_video is None.")
    print("get meta information of input video.")
    mapper, f_id, imgs_stack, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay = reset_video(input_video)
    EVE, EVE_config, processor, auto_masker, prompt_predictor = init_EVE()
    return EVE, EVE_config, processor, auto_masker, prompt_predictor, \
        mapper, f_id, imgs_stack, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay



@torch.no_grad()
def init_EVE(vid_length=20):  # TODO:想办法每次都输入vid_length
    from model.model_sam.network_sam import EVE
    from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    from segment_anything import SamPredictor
    config = edict({
        'model': 'saves/vith_full_s2_160000.pth',
        'd16_path': '/DAVIS-2017-trainval-480p',
        'd17_path': '/DAVIS-2017-trainval-480p',
        'y18_path': '/YouTube2018',
        'y19_path': '/YouTube',
        'lv_path': '/long_video_set',
        'generic_path': None,
        'dataset': 'D17',
        'split': 'val',
        'output': 'output/D17_val_xemsam',
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

    auto_masker = SamAutomaticMaskGenerator(network.sam)
    prompt_predictor = SamPredictor(network.sam)
    processor = InferenceCore(network, config=config)
    print("Initialize successfully.")
    return network, config, processor, auto_masker, prompt_predictor


def select_sorted(masks, allow_min_area=100, max_num_obj=20):
    if len(masks) == 0:
        return []
    masks = [m for m in masks if m['area'] > allow_min_area]
    masks = sorted(masks,
                   key=lambda dic: dic['predicted_iou'] * dic['stability_score'],
                   reverse=True)[:max_num_obj]
    return masks


@torch.no_grad()
def generate_masks_tensor(masks):
    masks_tensor = []
    for mask in masks:
        if isinstance(mask, dict):
            masks_tensor.append(torch.Tensor(mask['segmentation']))
        else:
            masks_tensor.append(torch.Tensor(mask[0]))
    masks_tensor = torch.stack(masks_tensor)
    bg_tensor = (torch.ones_like(masks_tensor[0]) - masks_tensor.sum(0))[None] == 1
    masks_tensor = torch.cat([bg_tensor, masks_tensor], 0)
    return masks_tensor


def check_overlay(masks_np, image, palette=PALETTE, save_overlay=False, f_id=None):
    masks_pil = Image.fromarray(masks_np.astype(np.uint8))
    masks_pil.putpalette(palette)
    masks_rgb = np.array(masks_pil.convert('RGB'))
    curr_overlay = cv2.addWeighted(image, 0.8, masks_rgb, 0.8, 0)
    if save_overlay and f_id is not None:  # save demo outputs
        overlay_path = "sam_scripts/demo_outputs/overlay"
        mask_path = "sam_scripts/demo_outputs/mask"
        if not os.path.exists(overlay_path):
            os.makedirs(overlay_path)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        cv2.imwrite(f"sam_scripts/demo_outputs/overlay/{str(f_id).zfill(5)}.png",
                    curr_overlay[:, :, ::-1])
        masks_pil.save(f"sam_scripts/demo_outputs/mask/{str(f_id).zfill(5)}.png")
    return curr_overlay


@torch.no_grad()
def seg_everything(image, auto_masker, ALLOW_MIN_AREA, MAX_NUM_OBJ):
    if image is None:
        raise gr.Error("No current frame. Please upload a video.")
    if auto_masker is None:
        raise gr.Error("No auto_masker. Please Init EVE first.")
    masks = auto_masker.generate(image)
    masks = select_sorted(masks, ALLOW_MIN_AREA, MAX_NUM_OBJ)
    masks_tensor = generate_masks_tensor(masks)  # [num_obj+bg, h, w]
    masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
    curr_overlay = check_overlay(masks_np, image, PALETTE)
    print("seg_everything on current frame successfully.")
    return masks_tensor, curr_overlay


def reform_box(box):
    # box:[[X0, Y0], [X1, Y1]]
    x0, y0 = box[0]
    x1, y1 = box[1]
    return [[min(x0, x1), min(y0, y1)],
            [max(x0, x1), max(y0, y1)]]


def add_clicks(prompt_predictor, curr_overlay, clicks_stack, point_mode, evt: gr.SelectData):
    if prompt_predictor is None:
        raise gr.Error("Please Init EVE first.")
    print(f"point_mode: {point_mode}")
    box = clicks_stack['box']
    point = clicks_stack['point']
    point_lb = clicks_stack['point_lb']
    if point_mode == "Single Box":
        box.append(evt.index)
        cv2.circle(curr_overlay, tuple(box[-1]), 6, (0, 255, 0), -1)
        if len(box) == 1:
            pass
        elif len(box) == 2:
            box = reform_box(box)
            cv2.rectangle(curr_overlay, tuple(box[0]), tuple(box[1]), (0, 255, 0), 2)
        elif len(box) == 3:
            box = box[2:]
    elif point_mode == "Positive Points":
        point.append(evt.index)
        point_lb.append(1)
        cv2.circle(curr_overlay, tuple(evt.index), 6, (0, 153, 255), -1)
    elif point_mode == "Negative Points":
        point.append(evt.index)
        point_lb.append(0)
        cv2.circle(curr_overlay, tuple(evt.index), 6, (255, 80, 80), -1)
    else:
        raise gr.Error("No such point mode.")
    clicks_stack['box'] = box
    clicks_stack['point'] = point
    clicks_stack['point_lb'] = point_lb
    print(f"clicks_stack: {clicks_stack}")
    return curr_overlay, clicks_stack


@torch.no_grad()
def cal_masks_tensor(prompt_predictor, clicks_stack):
    num_point = len(clicks_stack['point'])
    num_box = len(clicks_stack['box'])
    input_point = np.array(clicks_stack['point']) if num_point != 0 else None
    input_label = np.array(clicks_stack['point_lb']) if num_point != 0 else None
    input_box = np.array(clicks_stack['box']).flatten() if num_box != 0 else None
    masks, _, _ = prompt_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    masks = masks[None].astype(np.uint8)
    masks_tensor_new = generate_masks_tensor(masks)  # [num_obj+bg, h, w]
    return masks_tensor_new


@torch.no_grad()
def seg_clicks(prompt_predictor, curr_img, masks_tensor, clicks_stack):
    if prompt_predictor is None:
        raise gr.Error("Please Init EVE first.")
    prompt_predictor.set_image(curr_img)
    masks_tensor_new = cal_masks_tensor(prompt_predictor, clicks_stack)  # [num_obj+bg, h, w]

    if masks_tensor is None:
        masks_tensor = masks_tensor_new
    else:
        # 最新的object覆盖一号位
        masks_tensor[0] *= masks_tensor_new[0]
        masks_tensor[1] = masks_tensor_new[1]
    masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
    curr_overlay = check_overlay(masks_np, curr_img, PALETTE)
    clicks_stack = dict(box=[], point=[], point_lb=[])
    print("seg one object on current frame successfully.")
    return masks_tensor, clicks_stack, curr_overlay


@torch.no_grad()
def add_new_obj_click(prompt_predictor, curr_img, masks_tensor, clicks_stack):
    if prompt_predictor is None:
        raise gr.Error("Please Init EVE first.")
    prompt_predictor.set_image(curr_img)
    masks_tensor_new = cal_masks_tensor(prompt_predictor, clicks_stack)

    if masks_tensor is None:
        masks_tensor = masks_tensor_new
    else:
        # 最新的object插在一号位
        masks_tensor[0] *= masks_tensor_new[0]
        masks_tensor = torch.cat((masks_tensor[0:1],
                                  masks_tensor_new[-1:],
                                  masks_tensor[1:]), dim=0)
    masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
    curr_overlay = check_overlay(masks_np, curr_img, PALETTE)
    clicks_stack = dict(box=[], point=[], point_lb=[])
    print("add one object on current frame successfully.")
    return masks_tensor, clicks_stack, curr_overlay


@torch.no_grad()
def reset_curr(curr_img):
    curr_overlay = curr_img
    masks_tensor = None
    clicks_stack = dict(box=[], point=[], point_lb=[])
    print("Reset current frame succeccfully.")
    return masks_tensor, clicks_stack, curr_overlay


def reset_video(input_video):
    if input_video is None:
        raise gr.Error("Please load video first.")
    print("Reset video succeccfully.")
    cap = cv2.VideoCapture(input_video)
    imgs_stack = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # [H, W, C]
            imgs_stack.append(frame)
        else:
            break
    cap.release()

    msks_stack = []
    curr_img = imgs_stack[0]  # first_frame
    masks_tensor = None
    clicks_stack = dict(box=[], point=[], point_lb=[])
    curr_overlay = imgs_stack[0]  # first_frame
    f_id = 0
    mapper = MaskMapper()
    return mapper, f_id, imgs_stack, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay


@torch.no_grad()
def resize_mask(mask, size=480):
    # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
    h, w = mask.shape[-2:]
    min_hw = min(h, w)
    return F.interpolate(mask, (int(h / min_hw * size), int(w / min_hw * size)),
                         mode='nearest')


def last_frame(processor,
               f_id, imgs_stack, msks_stack):
    if f_id == 0:
        raise gr.Error("It is 1st frame now.")
    """找到历史的结果"""
    msk_device = processor.network.sam.device
    f_id -= 1
    curr_img = imgs_stack[f_id]
    masks_np = msks_stack[f_id]
    """调整输出"""
    curr_overlay = check_overlay(masks_np, curr_img, PALETTE)
    clicks_stack = dict(box=[], point=[], point_lb=[])
    masks_tensor = F.one_hot(torch.tensor(masks_np, device=msk_device)).permute(2, 0, 1)
    print("It is the {}th frame.".format(f_id))
    return processor, \
        f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay


# TODO:last_frame调整输入


@torch.no_grad()
def run_processor(processor, rgb, msk, labels, target_size):
    rgb = preprocess(TF.resize(rgb, target_size), 1024)
    if msk is not None:
        msk = preprocess(TF.resize(msk, target_size), 1024)
    prob = processor.step(rgb, msk, labels, end=False)
    return prob, processor


@torch.no_grad()
def next_frame(mapper, processor,
               f_id, imgs_stack, msks_stack, masks_tensor,
               save_overlay=False):
    vid_length = len(imgs_stack)
    msk_device = processor.network.sam.device
    if f_id < vid_length - 1:
        f_id += 1
    else:
        print("This is the last frame.")
        curr_img = imgs_stack[f_id]
        masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
        curr_overlay = check_overlay(masks_np, curr_img, PALETTE)
        clicks_stack = dict(box=[], point=[], point_lb=[])
        return processor, \
            f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay

    """如果已有结果就直接返回"""
    if len(msks_stack) > f_id:
        curr_img = imgs_stack[f_id]
        masks_np = msks_stack[f_id]
        curr_overlay = check_overlay(masks_np, curr_img, PALETTE, save_overlay, f_id)
        clicks_stack = dict(box=[], point=[], point_lb=[])
        masks_tensor = F.one_hot(torch.tensor(masks_np, device=msk_device)).permute(2, 0, 1)
        print("It is the {}th frame.".format(f_id))
        return processor, \
            f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay

    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53], device=msk_device).view(3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375], device=msk_device).view(3, 1, 1)
    input_size = imgs_stack[0].shape[0:2]
    target_size = get_preprocess_shape(*input_size, 1024)
    if f_id - 1 == 0:  # 第一帧，回顾第零帧信息
        """调整参考帧信息"""
        curr_img = imgs_stack[f_id]
        masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
        msk, labels = mapper.convert_mask(masks_np)
        msk = torch.Tensor(msk).to(msk_device)
        msk = resize_mask(msk[None])[0]
        processor.set_all_labels(list(mapper.remappings.values()))
        """记录参考帧信息,仅为更新processor"""
        rgb = torch.tensor(imgs_stack[f_id - 1]).permute(2, 0, 1).to(msk_device)
        rgb = (rgb - PIXEL_MEAN) / PIXEL_STD
        masks_tensor, processor = run_processor(processor, rgb, msk, labels, target_size)
        masks_tensor = postprocess_masks(masks_tensor, target_size, input_size)
        masks_tensor = F.interpolate(masks_tensor.unsqueeze(1), input_size,
                                     mode='bilinear', align_corners=False)[:, 0]
        masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]
        curr_overlay = check_overlay(masks_np, curr_img, PALETTE, save_overlay, f_id)
        msks_stack.append(masks_np)
    curr_img = imgs_stack[f_id]
    rgb = torch.tensor(curr_img).permute(2, 0, 1).to(msk_device)
    rgb = (rgb - PIXEL_MEAN) / PIXEL_STD
    masks_tensor, processor = run_processor(processor, rgb, None, None, target_size)
    masks_tensor = postprocess_masks(masks_tensor, target_size, input_size)
    masks_tensor = F.interpolate(masks_tensor.unsqueeze(1), input_size,
                                 mode='bilinear', align_corners=False)[:, 0]
    masks_np = torch.argmax(masks_tensor, dim=0).detach().cpu().numpy()  # [h, w]

    """调整输出"""
    curr_overlay = check_overlay(masks_np, curr_img, PALETTE, save_overlay, f_id)
    msks_stack.append(masks_np)
    clicks_stack = dict(box=[], point=[], point_lb=[])
    print(f"inference the No.{f_id} frame successfully.")
    return processor, \
        f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay


# 读取文件夹内所有图片，并用cv2合成mp4格式的视频
def picvideo(path, mp4_path):
    filelist = os.listdir(path)  # 获取该目录下的所有文件名
    filelist.sort(key=lambda x: int(x[:-4]))
    fps = 30.0  # 视频每秒帧数
    size = cv2.imread(path + '/' + filelist[0]).shape[:2][::-1]  # [W,H]
    fourcc_type = 'mp4v'  # 'avc1'  # 'x264'  # 'H264'  'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)  # 视频格式
    video = cv2.VideoWriter(mp4_path, fourcc, fps, size)
    for item in tqdm(filelist):
        if item.endswith('.png'):
            item = path + '/' + item
            img = cv2.imread(item)  # .astype(np.float)
            video.write(img)
    video.release()
    print(f'save output_video in "{mp4_path}" successfully.')


# 压缩文件夹成zip文件
def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    for parent, dirnames, filenames in source_dirs:
        for filename in filenames:
            if filename.endswith('.png'):
                pathfile = os.path.join(parent, filename)
                arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
                zipf.write(pathfile, arcname)
    zipf.close()
    print('save output_zip successfully.')


@torch.no_grad()
def run_all(mapper, processor,
            f_id, imgs_stack, msks_stack, masks_tensor):
    # 输入状态应为第零帧，但是已经完成了第零帧的标注，
    # processor经过处理，masks_tensor不为空
    if f_id != 0:
        raise gr.Error("Run all should be called at 1st frame.")
    if masks_tensor is None:
        raise gr.Error("Run all need the reference mask of 1st frame.")
    if len(imgs_stack) == 0:
        raise gr.Error("Run all need that num_frames > 1.")
    next_frame_input = [mapper, processor,
                        f_id, imgs_stack, msks_stack, masks_tensor]
    vid_length = len(imgs_stack)
    while f_id < vid_length - 1:
        next_frame_output = next_frame(*next_frame_input, save_overlay=True)
        processor, \
            f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay = next_frame_output
        next_frame_input = [mapper, processor,
                            f_id, imgs_stack, msks_stack, masks_tensor]

    """整理本地文件"""
    output_video = "sam_scripts/demo_outputs/video.mp4"
    output_overlay = "sam_scripts/demo_outputs/overlay.zip"
    output_mask = "sam_scripts/demo_outputs/mask.zip"
    picvideo("sam_scripts/demo_outputs/overlay", output_video)
    make_zip("sam_scripts/demo_outputs/overlay", output_overlay)
    shutil.rmtree("sam_scripts/demo_outputs/overlay")
    make_zip("sam_scripts/demo_outputs/mask", output_mask)
    shutil.rmtree("sam_scripts/demo_outputs/mask")

    """调整输出"""
    curr_img = imgs_stack[f_id]
    clicks_stack = dict(box=[], point=[], point_lb=[])

    return processor, \
        f_id, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay, \
        output_video, output_overlay, output_mask


if __name__ == '__main__':
    pass
    # data_next_frame = torch.load("sam_scripts/assets/next_frame.pth")
    # EVE, EVE_config, mapper, processor, auto_masker, prompt_predictor, \
    #     f_id, imgs_stack, msks_stack, curr_img, masks_tensor, clicks_stack, curr_overlay = data_next_frame
    #
    # next_frame(EVE, mapper, processor,
    #            f_id, msks_stack, masks_tensor)

    path = "sam_scripts/demo_outputs/overlay"
    mp4_path = "sam_scripts/demo_outputs/video.mp4"
    picvideo(path, mp4_path)
