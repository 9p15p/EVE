#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 19:31
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : sam_on_DAVIS.py
# @Describe: Provide bboxes for SAM to perform video object segmentation on DAVIS.
import os
import torch
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from shutil import copy
from tqdm import tqdm
import pandas as pd
from time import time
import sys
sys.path.append(os.getcwd())

from davis2017evaluation.davis2017.evaluation import DAVISEvaluation
from segment_anything import sam_model_registry, SamPredictor
from util.utils import masks_to_boxes
from util.plot_save import fvis


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


def get_all_file_paths(folder_path):
    """
    返回文件夹下所有jpg和png文件的路径
    :param folder_path: 文件夹路径
    :return: jpg_paths, png_paths
    """
    jpg_paths = []
    val_set = "/data/ldz_proj/DATASET/DAVIS2017/ImageSets/2017/val.txt"
    with open(val_set, 'r') as f:
        vals = f.readlines()
        vals = [val[:-1] for val in vals]
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if '.jpg' in file and root.split('/')[-1] in vals:
                jpg_paths.append(os.path.abspath(os.path.join(root, file)))
    png_paths = []
    for jpg in jpg_paths:
        png_paths.append(jpg.replace('.jpg', '.png').replace('JPEGImages', 'Annotations'))
    return jpg_paths, png_paths


# 读取图片
def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_msk(path):
    msk = Image.open(path).convert('P')
    msk = np.array(msk)
    return msk


def one_hot(msk):
    """msk的shape为(h,w)"""
    msk = torch.from_numpy(msk)
    msk = F.one_hot(msk.long())
    msk = msk.permute(2, 0, 1)
    return msk


# 如果没有该文件夹，则创建
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


if __name__ == '__main__':
    sam_checkpoint = "saves/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    palette = Image.open("/data/ldz_proj/DATASET/DAVIS2017/Annotations/480p/bike-packing/00000.png").getpalette()
    DAVIS_path = "/data/ldz_proj/DATASET/DAVIS2017"
    jpg_paths, png_paths = get_all_file_paths(DAVIS_path)
    for jpg_path, png_path in tqdm(zip(jpg_paths, png_paths)):
        if '00000' in jpg_path:
            copy(png_path, f"output/sam_on_DAVIS/{png_path.split('/')[-2]}/")
            continue
        img = read_img(jpg_path)
        msk = read_msk(png_path)
        msk_oh = one_hot(msk).to(device)
        bboxs = masks_to_boxes(msk_oh[1:])
        predictor.set_image(img)

        transformed_boxes = predictor.transform.apply_boxes_torch(bboxs, img.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        tmp = masks.permute(1, 0, 2, 3)
        tmp = torch.cat([torch.zeros_like(tmp[:, :1]), tmp], dim=1)
        tmp = torch.argmax(tmp.to(torch.uint8), dim=1)[0].detach().cpu()
        out_msk = Image.fromarray(np.uint8(tmp))
        out_msk.putpalette(palette)
        v_name = jpg_path.split('/')[-2]
        mkdir(f"output/sam_on_DAVIS/{v_name}")
        out_msk.save(f"output/sam_on_DAVIS/{v_name}/{png_path.split('/')[-1]}")

    split = 'val'
    time_start = time()
    print(f'Evaluating sequences for the semi-supervised task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=DAVIS_path, task='semi-supervised', gt_set=split)
    metrics_res = dataset_eval.evaluate("output/sam_on_DAVIS")
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array(
        [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
         np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {split} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {split} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    sys.stdout.write('\nTotal evaluation_time:' + str(total_time))
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in bboxs:
        #     show_box(box.cpu().numpy(), plt.gca())
        # plt.axis('off')
        # plt.show()

        # for i in range(len(bboxs)):
        #     masks, _, _ = predictor.predict(
        #         point_coords=None,
        #         point_labels=None,
        #         box=np.array(bboxs[i:i+1].detach().cpu()),
        #         multimask_output=False,
        #     )
        #
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(img)
        #     show_mask(masks[0], plt.gca())
        #     show_box(bboxs[i].detach().cpu(), plt.gca())
        #     plt.axis('off')
        #     plt.show()

        # print("hello")
