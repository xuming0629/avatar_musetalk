#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3FD 推理与后处理。

关键修复：
- S3FD 在部分 torch/cuDNN 环境下，CUDA forward 使用 cuDNN Conv2d 会触发段错误；
- 这里在 S3FD forward 期间临时关闭 cuDNN；
- 注意：关闭 cuDNN 不等于使用 CPU，模型和输入仍然在 CUDA 上执行。
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.nets.face.s3fd.bbox import decode, batch_decode


S3FD_MEAN_BGR = np.array([104, 117, 123], dtype=np.float32)


def _to_device(device):
    """
    将字符串或 torch.device 统一转换成 torch.device。
    """
    if isinstance(device, torch.device):
        return device

    device = str(device)

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[S3FD detect] CUDA not available, fallback to CPU")
        device = "cpu"

    return torch.device(device)


def _forward_without_cudnn(net, inputs, device):
    """
    S3FD CUDA forward 专用封装。

    说明：
    - 如果 device 是 cuda，则临时关闭 cuDNN，避开当前环境下的 cuDNN 段错误；
    - 关闭 cuDNN 后仍然走 CUDA，不会回退到 CPU；
    - 使用 try/finally 确保异常时也能恢复原始 cuDNN 设置。
    """
    if device.type != "cuda":
        with torch.inference_mode():
            return net(inputs)

    old_cudnn_enabled = torch.backends.cudnn.enabled
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic

    try:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.cuda.synchronize()

        with torch.inference_mode():
            outputs = net(inputs)

        torch.cuda.synchronize()
        return outputs

    finally:
        torch.backends.cudnn.enabled = old_cudnn_enabled
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic


def detect(net, img, device):
    """
    单张图片 S3FD 检测。

    保持原接口不变：
        detect(net, img, device)

    Args:
        net: S3FD 网络；
        img: np.ndarray，BGR 图片；
        device: "cpu" / "cuda" / torch.device。

    Returns:
        np.ndarray, shape=[N, 5]，每行为 [x1, y1, x2, y2, score]。
    """
    device = _to_device(device)

    if img is None:
        raise ValueError("Input image is None")

    img = img.astype(np.float32)
    img = img - S3FD_MEAN_BGR
    img = img.transpose(2, 0, 1).copy()
    img = img.reshape((1,) + img.shape)

    img = torch.from_numpy(img).float().to(device)

    net.eval()
    olist = _forward_without_cudnn(net, img, device)

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    olist = [oelem.detach().cpu() for oelem in olist]

    bboxlist = []

    for i in range(len(olist) // 2):
        ocls = olist[i * 2]
        oreg = olist[i * 2 + 1]

        stride = 2 ** (i + 2)
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))

        for _, hindex, windex in poss:
            axc = stride / 2 + windex * stride
            ayc = stride / 2 + hindex * stride

            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)

            priors = torch.tensor(
                [[axc, ayc, stride * 4, stride * 4]],
                dtype=torch.float32,
            )

            box = decode(
                loc,
                priors,
                [0.1, 0.2],
            )

            x1, y1, x2, y2 = box[0]

            bboxlist.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                ]
            )

    if len(bboxlist) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(bboxlist, dtype=np.float32)


def batch_detect(net, imgs, device):
    """
    批量图片 S3FD 检测。

    保持原接口不变：
        batch_detect(net, imgs, device)

    Args:
        net: S3FD 网络；
        imgs: np.ndarray, shape=[B, H, W, 3]，BGR 图片；
        device: "cpu" / "cuda" / torch.device。

    Returns:
        np.ndarray, shape=[K, B, 5]。
    """
    device = _to_device(device)

    if imgs is None:
        raise ValueError("Input images is None")

    imgs = imgs.astype(np.float32)
    imgs = imgs - S3FD_MEAN_BGR
    imgs = imgs.transpose(0, 3, 1, 2).copy()

    imgs = torch.from_numpy(imgs).float().to(device)

    net.eval()
    olist = _forward_without_cudnn(net, imgs, device)

    batch_size = imgs.size(0)

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    olist = [oelem.detach().cpu() for oelem in olist]

    bboxlist = []

    for i in range(len(olist) // 2):
        ocls = olist[i * 2]
        oreg = olist[i * 2 + 1]

        stride = 2 ** (i + 2)
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))

        for _, hindex, windex in poss:
            axc = stride / 2 + windex * stride
            ayc = stride / 2 + hindex * stride

            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(batch_size, 1, 4)

            priors = torch.tensor(
                [[axc, ayc, stride * 4, stride * 4]],
                dtype=torch.float32,
            ).view(1, 1, 4)

            box = batch_decode(
                loc,
                priors,
                [0.1, 0.2],
            )

            box = box[:, 0]

            bboxlist.append(
                torch.cat(
                    [
                        box,
                        score.unsqueeze(1),
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )

    if len(bboxlist) == 0:
        return np.zeros((0, batch_size, 5), dtype=np.float32)

    return np.array(bboxlist, dtype=np.float32)


def flip_detect(net, img, device):
    """
    水平翻转图片后进行检测，并将框映射回原坐标系。
    """
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape, dtype=np.float32)

    if len(b) == 0:
        return bboxlist

    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]

    return bboxlist


def pts_to_bb(pts):
    """
    根据关键点坐标生成外接矩形。
    """
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)

    return np.array([min_x, min_y, max_x, max_y], dtype=np.float32)
