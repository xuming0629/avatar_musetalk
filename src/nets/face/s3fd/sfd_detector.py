#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : sfd_detector.py
# @Time          : 2026-06-23 22:56:44
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : 支持配置文件的 S3FD 人脸检测器
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import os
from typing import Optional

import torch
from torch.utils.model_zoo import load_url

from src.nets.common.config import load_yaml, resolve_path, get_device
from src.nets.face.s3fd.face_detect import FaceDetector

from .bbox import *
from .detect import *
from .s3fd_net import s3fd


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"

models_urls = {
    "s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
}


def get_s3fd_path(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
) -> str:
    """从配置文件读取 S3FD 权重路径。"""
    cfg = load_yaml(config_path)
    model_path = cfg["models"]["s3fd"]["path"]
    return resolve_path(model_path, project_root)


class SFDDetector(FaceDetector):
    """S3FD 人脸检测器。

    保持原有调用方式：
        SFDDetector()
        SFDDetector.from_config(...)
        SFDDetector(device="cuda", path_to_detector="xxx.pth")
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device=None,
        path_to_detector: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        if device is None:
            device = get_device(config_path=config_path)

        if path_to_detector is None:
            path_to_detector = get_s3fd_path(
                config_path=config_path,
                project_root=project_root,
            )

        super().__init__(device, verbose)

        self.config_path = config_path
        self.project_root = project_root
        self.device = str(device)
        self.path_to_detector = path_to_detector

        if not os.path.isfile(path_to_detector):
            if not auto_download:
                raise FileNotFoundError(
                    f"S3FD detector weight not found: {path_to_detector}"
                )

            print(f"[SFDDetector] model not found, download from: {models_urls['s3fd']}")
            model_weights = load_url(models_urls["s3fd"], map_location="cpu")

        else:
            try:
                model_weights = torch.load(
                    path_to_detector,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # 兼容不支持 weights_only 参数的旧版 PyTorch
                model_weights = torch.load(path_to_detector, map_location="cpu")

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights, strict=True)
        self.face_detector.to(self.device)
        self.face_detector.eval()

        if "cuda" in self.device:
            torch.cuda.synchronize()

        print(f"[SFDDetector] load weight success: {path_to_detector}")

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        """从配置文件创建检测器，兼容旧代码。"""
        return cls(
            config_path=config_path,
            project_root=project_root,
            device=None,
            path_to_detector=None,
            verbose=verbose,
            auto_download=auto_download,
        )

    def detect_from_image(self, tensor_or_path):
        """检测单张图片，并执行 NMS 与置信度过滤。"""
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(
            self.face_detector,
            image,
            device=self.device,
        )

        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]

        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        """批量检测图片，并对每张图片执行 NMS 与置信度过滤。"""
        bboxlists = batch_detect(
            self.face_detector,
            images,
            device=self.device,
        )

        keeps = [
            nms(bboxlists[:, i, :], 0.3)
            for i in range(bboxlists.shape[1])
        ]

        bboxlists = [
            bboxlists[keep, i, :]
            for i, keep in enumerate(keeps)
        ]

        bboxlists = [
            [x for x in bboxlist if x[-1] > 0.5]
            for bboxlist in bboxlists
        ]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
