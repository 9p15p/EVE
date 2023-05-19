#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 14:22
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : concat_img_to_video.py
# @Describe:

from argparse import ArgumentParser
import os
from easydict import EasyDict as edict
import cv2
from tqdm import tqdm
from util.plot_save import fvis


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--vid_path', type=str, default="sam_scipts/assets/")
    config = edict(vars(parser.parse_args()))
    assert config.img_dir is not None, 'img_dir is None'
    config.vid_path = os.path.join(config.vid_path, os.path.basename(config.img_dir) + '.mp4')
    return config


def concat_img_to_video(img_dir: str, vid_path: str):
    assert os.path.exists(img_dir), f'{img_dir} not exists'
    assert os.path.exists(os.path.dirname(vid_path)), f'{os.path.dirname(vid_path)} not exists'

    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda x: int(x.split('.')[0]))
    img_list = [os.path.join(img_dir, img) for img in img_list]
    # print(img_list)
    img = cv2.imread(img_list[0])
    height, width, layers = img.shape
    fourcc_type = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)
    video = cv2.VideoWriter(vid_path, fourcc, 30, (width, height))
    for img in tqdm(img_list):
        video.write(cv2.imread(img))
    # cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    # config = get_config()
    # concat_img_to_video(config.img_dir, config.vid_path)

    img_dirs = [
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/DAVIS2017/JPEGImages/480p/breakdance",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/DAVIS2017/JPEGImages/480p/lab-coat",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/DAVIS2017/JPEGImages/480p/dogs-jump",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/3f99366076",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/7775043b5e",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/19904980af",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/8273b59141",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/53edbc6b8a",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/c74fc37224",
        "/media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/DATASET/YoutubeVOS2019/valid/JPEGImages/1ab5f4bbc5",

    ]
    for img_dir in img_dirs:
        vid_path = os.path.join("sam_scripts/assets/", os.path.basename(img_dir) + '.mp4')
        concat_img_to_video(img_dir, vid_path)

