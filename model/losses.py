import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from util.plot_save import fvis
from collections import defaultdict
from torchvision.ops import roi_align
from util.utils import masks_to_boxes, draw_line

def dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:, i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt == (i + 1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])

    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi + 1, :num_objects[bi] + 1],
                                   data['cls_gt'][bi:bi + 1, ti, 0], it)
                losses['p'] += p / b / (t - 1)
                losses[f'ce_loss_{ti}'] += loss / b

            losses['total_loss'] += losses['ce_loss_%d' % ti]
            losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:, ti, 0])
            losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses

    def compute_cc(self, data, num_objects, it):
        assert self.config['use_cc'], "must activate --use_cc in args"
        losses = defaultdict(int)
        b, t = data['rgb'].shape[:2]
        for ti in range(1, t):
            t_softmax = torch.softmax(data[f'logits_{ti}'], 1)
            s_softmax = torch.softmax(data[f'logits_{ti}_t'], 1)
            losses[f'mse_loss_{ti}'] += F.mse_loss(s_softmax, t_softmax)
            losses['total_loss'] += losses['mse_loss_%d' % ti]
        return losses

    def compute_contrast(self, data, num_objects, it):
        assert self.config['use_contrast'], "must activate --use_contrast in args"
        losses = defaultdict(int)
        losses['total_loss'] = 0
        b, t = data['rgb'].shape[:2]

        # student
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi + 1, :num_objects[bi] + 1],
                                   data['cls_gt_from_t'][bi:bi + 1, ti, 0], it)
                losses['p_s'] += p / b / (t - 1)
                losses[f'ce_{ti}_s'] += loss / b

            losses['total_s'] += losses['ce_%d_s' % ti]
            losses[f'dice_{ti}_s'] = dice_loss(data[f'masks_{ti}'], data['cls_gt_from_t'][:, ti, 0])
            losses['total_s'] += losses[f'dice_{ti}_s']

        # teacher
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}_t'][bi:bi + 1, :num_objects[bi] + 1],
                                   data['cls_gt_from_s'][bi:bi + 1, ti, 0], it)
                losses['p_t'] += p / b / (t - 1)
                losses[f'ce_{ti}_t'] += loss / b

            losses['total_t'] += losses['ce_%d_t' % ti]
            losses[f'dice_{ti}_t'] = dice_loss(data[f'masks_{ti}_t'], data['cls_gt_from_s'][:, ti, 0])
            losses['total_t'] += losses[f'dice_{ti}_t']

        losses['total_loss'] += losses['total_s'] + losses['total_t']

        return losses


