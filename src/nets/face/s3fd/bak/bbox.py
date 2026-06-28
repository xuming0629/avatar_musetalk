#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3FD 边界框工具函数。

包含：
1. bbox 编码/解码；
2. NMS 非极大值抑制；
3. 可选 IOU 加速接口。

说明：
- 保持原函数接口不变；
- decode / batch_decode 用于 S3FD 检测结果还原；
- nms 返回保留框的索引列表。
"""

from __future__ import print_function

import math
import numpy as np
import torch

try:
    from iou import IOU
except BaseException:
    # 如果没有 cython 版 IOU，则使用 Python 版兜底。
    def IOU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        sa = abs((ax2 - ax1) * (ay2 - ay1))
        sb = abs((bx2 - bx1) * (by2 - by1))

        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)

        w = x2 - x1
        h = y2 - y1

        if w < 0 or h < 0:
            return 0.0

        return 1.0 * w * h / (sa + sb - w * h)


def bboxlog(x1, y1, x2, y2, axc, ayc, aww, ahh):
    """
    将真实框编码成相对 anchor 的偏移量。
    """
    xc = (x2 + x1) / 2
    yc = (y2 + y1) / 2
    ww = x2 - x1
    hh = y2 - y1

    dx = (xc - axc) / aww
    dy = (yc - ayc) / ahh
    dw = math.log(ww / aww)
    dh = math.log(hh / ahh)

    return dx, dy, dw, dh


def bboxloginv(dx, dy, dw, dh, axc, ayc, aww, ahh):
    """
    将相对 anchor 的偏移量还原成 xyxy 边界框。
    """
    xc = dx * aww + axc
    yc = dy * ahh + ayc
    ww = math.exp(dw) * aww
    hh = math.exp(dh) * ahh

    x1 = xc - ww / 2
    x2 = xc + ww / 2
    y1 = yc - hh / 2
    y2 = yc + hh / 2

    return x1, y1, x2, y2


def nms(dets, thresh):
    """
    非极大值抑制。

    Args:
        dets: np.ndarray, shape=[N, 5], 每行为 [x1, y1, x2, y2, score]
        thresh: float, IOU 阈值

    Returns:
        keep: list[int], 保留框索引
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def encode(matched, priors, variances):
    """
    将匹配到的真实框编码成 S3FD 回归目标。

    Args:
        matched: Tensor, shape=[num_priors, 4], xyxy 格式真实框
        priors: Tensor, shape=[num_priors, 4], [cx, cy, w, h]
        variances: list[float], 编码方差

    Returns:
        Tensor, shape=[num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy = g_cxcy / (variances[0] * priors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], dim=1)


def decode(loc, priors, variances):
    """
    将 S3FD 回归输出解码成 xyxy 边界框。

    Args:
        loc: Tensor, shape=[num_priors, 4]
        priors: Tensor, shape=[num_priors, 4], [cx, cy, w, h]
        variances: list[float]

    Returns:
        Tensor, shape=[num_priors, 4], xyxy 格式
    """
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        dim=1,
    )

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def batch_decode(loc, priors, variances):
    """
    batch 版本 decode。

    Args:
        loc: Tensor, shape=[B, num_priors, 4]
        priors: Tensor, shape=[B, num_priors, 4]
        variances: list[float]

    Returns:
        Tensor, shape=[B, num_priors, 4], xyxy 格式
    """
    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )

    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]

    return boxes
