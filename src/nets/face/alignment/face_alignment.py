#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : face_alignment.py
# @Time          : 2026-06-24 10:38:27
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : FaceAlignment wrapper with default config support
# @Company       : 2026 XuMing. All Rights Reserved.
"""

from enum import Enum
from typing import Optional, Union, List, Tuple

import numpy as np

from src.nets.face.s3fd.sfd_detector import SFDDetector


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"


class LandmarksType(Enum):
    """
    Enum class defining the type of landmarks to detect.

    _2D:
        detected points (x, y) in 2D space

    _2halfD:
        projected 3D points

    _3D:
        detected points (x, y, z) in 3D space
    """

    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """
    FaceAlignment wrapper.

    默认支持直接声明：

        face_alignment = FaceAlignment()

    等价于：

        face_alignment = FaceAlignment.from_config("configs/musetalk_v15.yaml")

    也支持外部传入 detector：

        detector = SFDDetector()
        face_alignment = FaceAlignment(detector=detector)
    """

    def __init__(
        self,
        detector: Optional[SFDDetector] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        """
        参数:
            detector:
                可选。外部已经创建好的 SFDDetector。

            config_path:
                默认配置路径。如果 detector 不传，则用这个配置创建 SFDDetector。

            project_root:
                项目根目录，用于 resolve_path。

            verbose:
                SFDDetector 日志开关。

            auto_download:
                SFD 权重不存在时是否自动下载。
        """

        self.config_path = config_path
        self.project_root = project_root

        if detector is None:
            detector = SFDDetector(
                config_path=config_path,
                project_root=project_root,
                verbose=verbose,
                auto_download=auto_download,
            )

        self.face_detector = detector

        print("[FaceAlignment] init success")

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        """
        保留 from_config 写法，兼容旧代码。
        """

        return cls(
            detector=None,
            config_path=config_path,
            project_root=project_root,
            verbose=verbose,
            auto_download=auto_download,
        )

    def get_detections_for_batch(
        self,
        images: np.ndarray,
    ) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        批量检测人脸 bbox。

        输入:
            images:
                - [H, W, C]
                - [B, H, W, C]

        输出:
            results:
                [
                    (x1, y1, x2, y2),
                    None,
                    ...
                ]
        """

        if images is None:
            raise ValueError("images is None")

        if not isinstance(images, np.ndarray):
            images = np.asarray(images)

        if images.ndim == 3:
            images = images[None, ...]

        if images.ndim != 4:
            raise ValueError(
                f"images ndim must be 3 or 4, got {images.ndim}, shape={images.shape}"
            )

        results = []

        for img in images:
            bboxes = self.face_detector.detect_from_image(
                img,
            )

            if bboxes is None or len(bboxes) == 0:
                results.append(None)
                continue

            # SFD 返回一般是 [x1, y1, x2, y2, score]
            # 默认取第一个检测框
            bbox = np.asarray(bboxes[0])

            if bbox.shape[0] < 4:
                results.append(None)
                continue

            x1, y1, x2, y2 = map(
                int,
                bbox[:4],
            )

            results.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

        return results

    def detect_from_batch(
        self,
        images: np.ndarray,
    ) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        兼容 MuseTalk 原始调用方式：

            fa.detect_from_batch(images)

        等价于：

            fa.get_detections_for_batch(images)
        """

        return self.get_detections_for_batch(
            images,
        )

    def detect_from_image(
        self,
        image: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        单张图片检测。

        返回:
            (x1, y1, x2, y2) 或 None
        """

        results = self.get_detections_for_batch(
            image,
        )

        if len(results) == 0:
            return None

        return results[0]


if __name__ == "__main__":
    import cv2

    face_alignment = FaceAlignment()

    img_path = "assets/sit.jpeg"
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"读取图片失败: {img_path}")

    bbox = face_alignment.detect_from_image(img)

    print("bbox:", bbox)

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        vis = img.copy()

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        save_path = "assets/sit_face_bbox.jpg"

        cv2.imwrite(
            save_path,
            vis,
        )

        print(f"save result to: {save_path}")