#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 14:40
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : eval_util.py
# @Describe:
import torch
import torch.nn.functional as F
from typing import Tuple


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def preprocess(x: torch.Tensor, img_size) -> torch.Tensor:
    """pad to a square input."""
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def postprocess_masks(
        masks: torch.Tensor,
        target_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
) -> torch.Tensor:
    # masks = F.interpolate(
    #     masks,
    #     (self.image_encoder.img_size, self.image_encoder.img_size),
    #     mode="bilinear",
    #     align_corners=False,
    # )
    masks = masks[..., : target_size[0], : target_size[1]]
    masks = F.interpolate(masks[None], original_size, mode="bilinear", align_corners=False)[0]
    return masks
