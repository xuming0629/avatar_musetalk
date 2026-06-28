#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : detect.py
# @Time          : 2026-06-23 22:57:49
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : S3FD 检测后处理与推理封装
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.nets.face.s3fd.bbox import decode, batch_decode


S3FD_MEAN_BGR = np.array([104, 117, 123])


def _is_cuda_device(device):
    """判断 device 是否为 CUDA 设备，兼容字符串和 torch.device。"""
    return "cuda" in str(device)


def _forward_s3fd(net, inputs, device):
    """执行 S3FD forward。

    说明：
        1. CPU 逻辑不变，仍然直接执行 net(inputs)。
        2. CUDA 逻辑仍然走 GPU，不回退 CPU。
        3. 仅在 S3FD forward 期间临时关闭 cuDNN，规避当前环境下
           cuDNN Conv2d 触发的 Segmentation fault。
        4. 使用 try/finally 保证异常时也恢复原来的 cuDNN 全局状态，
           避免影响后续 VAE、UNet、RTMPose 等模型。
    """
    if not _is_cuda_device(device):
        with torch.no_grad():
            return net(inputs)

    old_cudnn_enabled = torch.backends.cudnn.enabled
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic

    try:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.cuda.synchronize()

        with torch.no_grad():
            outputs = net(inputs)

        torch.cuda.synchronize()
        return outputs

    finally:
        torch.backends.cudnn.enabled = old_cudnn_enabled
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic


def detect(net, img, device):
    """单张图片人脸检测。

    参数:
        net: S3FD 网络。
        img: 输入图片，保持原始逻辑，按 HWC BGR/RGB 数组进入。
        device: "cpu"、"cuda"、"cuda:0" 或 torch.device。

    返回:
        numpy.ndarray，形状为 [N, 5]，每行为 [x1, y1, x2, y2, score]。

    注意:
        本函数保持原有预处理、阈值、decode、返回结构不变。
    """
    img = img - S3FD_MEAN_BGR
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    img = torch.from_numpy(img).float().to(device)
    BB, CC, HH, WW = img.size()

    olist = _forward_s3fd(net, img, device)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    olist = [oelem.data.cpu() for oelem in olist]

    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()
        stride = 2 ** (i + 2)
        anchor = stride * 4

        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc = stride / 2 + windex * stride
            ayc = stride / 2 + hindex * stride

            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)

            priors = torch.Tensor(
                [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
            )
            variances = [0.1, 0.2]

            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0

            bboxlist.append([x1, y1, x2, y2, score])

    bboxlist = np.array(bboxlist)

    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))

    return bboxlist


def batch_detect(net, imgs, device):
    """批量图片人脸检测。

    参数:
        net: S3FD 网络。
        imgs: 输入图片批次，形状为 [B, H, W, C]。
        device: "cpu"、"cuda"、"cuda:0" 或 torch.device。

    返回:
        numpy.ndarray，形状保持原始逻辑。
    """
    imgs = imgs - S3FD_MEAN_BGR
    imgs = imgs.transpose(0, 3, 1, 2)

    imgs = torch.from_numpy(imgs).float().to(device)
    BB, CC, HH, WW = imgs.size()

    olist = _forward_s3fd(net, imgs, device)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    olist = [oelem.cpu() for oelem in olist]

    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()
        stride = 2 ** (i + 2)
        anchor = stride * 4

        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc = stride / 2 + windex * stride
            ayc = stride / 2 + hindex * stride

            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(BB, 1, 4)

            priors = torch.Tensor(
                [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]
            ).view(1, 1, 4)
            variances = [0.1, 0.2]

            box = batch_decode(loc, priors, variances)
            box = box[:, 0] * 1.0

            bboxlist.append(
                torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy()
            )

    bboxlist = np.array(bboxlist)

    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))

    return bboxlist


def flip_detect(net, img, device):
    """水平翻转图片后检测，并将检测框映射回翻转坐标。"""
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]

    return bboxlist


def pts_to_bb(pts):
    """由关键点坐标计算外接矩形。"""
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    return np.array([min_x, min_y, max_x, max_y])
