#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3FD 人脸检测器封装。

说明：
- 保持原有类名、函数名和入参不变；
- 支持从配置文件读取模型路径和运行设备；
- S3FD 使用 BGR 输入，因此 detect_from_image 内部显式 rgb=False；
- CUDA 场景下的 cuDNN 段错误修复在 detect.py 内完成。
"""

import os
from typing import Optional

import torch
from torch.utils.model_zoo import load_url

from src.nets.common.config import load_yaml, resolve_path, get_device
from src.nets.face.s3fd.face_detect import FaceDetector

from .bbox import nms
from .detect import batch_detect, detect
from .s3fd_net import s3fd


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"

models_urls = {
    "s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
}


def _safe_device(device):
    """
    规范化 device。

    说明：
    - 不改变外部接口；
    - 如果请求 CUDA 但当前不可用，则自动回退 CPU。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = str(device)

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[SFDDetector] CUDA not available, fallback to CPU")
        device = "cpu"

    return device


def get_s3fd_path(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
) -> str:
    """
    从配置文件中读取 S3FD 权重路径。
    """
    cfg = load_yaml(config_path)
    model_path = cfg["models"]["s3fd"]["path"]

    return resolve_path(
        model_path,
        project_root,
    )


class SFDDetector(FaceDetector):
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
            device = get_device(
                config_path=config_path,
            )

        device = _safe_device(device)

        if path_to_detector is None:
            path_to_detector = get_s3fd_path(
                config_path=config_path,
                project_root=project_root,
            )

        super().__init__(
            device,
            verbose,
        )

        self.config_path = config_path
        self.project_root = project_root
        self.device = device
        self.path_to_detector = path_to_detector

        print("[SFDDetector] use device:", self.device)

        model_weights = self._load_weights(
            path_to_detector=path_to_detector,
            auto_download=auto_download,
        )

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(
            model_weights,
            strict=True,
        )

        self.face_detector.to(self.device)
        self.face_detector.eval()

        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

        print(f"[SFDDetector] load weight success: {path_to_detector}")

    def _load_weights(
        self,
        path_to_detector: str,
        auto_download: bool = False,
    ):
        """
        加载 S3FD 权重。
        """
        if not os.path.isfile(path_to_detector):
            if not auto_download:
                raise FileNotFoundError(
                    f"S3FD detector weight not found: {path_to_detector}"
                )

            print(f"[SFDDetector] model not found, download from: {models_urls['s3fd']}")
            return load_url(
                models_urls["s3fd"],
                map_location="cpu",
            )

        try:
            return torch.load(
                path_to_detector,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(
                path_to_detector,
                map_location="cpu",
            )

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        return cls(
            config_path=config_path,
            project_root=project_root,
            device=None,
            path_to_detector=None,
            verbose=verbose,
            auto_download=auto_download,
        )

    def detect_from_image(self, tensor_or_path):
        """
        单张图片检测。

        Returns:
            list[np.ndarray]，每个元素为 [x1, y1, x2, y2, score]。
        """
        image = self.tensor_or_path_to_ndarray(
            tensor_or_path,
            rgb=False,
        )

        bboxlist = detect(
            self.face_detector,
            image,
            device=self.device,
        )

        if bboxlist is None or len(bboxlist) == 0:
            return []

        keep = nms(
            bboxlist,
            0.3,
        )

        bboxlist = bboxlist[
            keep,
            :,
        ]

        return [
            x for x in bboxlist
            if x[-1] > 0.5
        ]

    def detect_from_batch(self, images):
        """
        批量图片检测。
        """
        bboxlists = batch_detect(
            self.face_detector,
            images,
            device=self.device,
        )

        if bboxlists is None or len(bboxlists) == 0:
            return [[] for _ in range(len(images))]

        keeps = [
            nms(
                bboxlists[:, i, :],
                0.3,
            )
            for i in range(bboxlists.shape[1])
        ]

        bboxlists = [
            bboxlists[
                keep,
                i,
                :,
            ]
            for i, keep in enumerate(keeps)
        ]

        return [
            [
                x for x in bboxlist
                if x[-1] > 0.5
            ]
            for bboxlist in bboxlists
        ]

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
