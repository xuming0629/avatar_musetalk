#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : sfd_detector.py
# @Author        : XuMing
# @description   : S3FD face detector with config support
"""

import os
from typing import Optional

import torch
from torch.utils.model_zoo import load_url

from src.nets.common.config import (
    load_yaml,
    resolve_path,
    get_device,
)

from src.nets.face.s3fd.face_detect import FaceDetector

from .s3fd_net import s3fd
from .bbox import *
from .detect import *


models_urls = {
    "s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
}


def get_s3fd_path(
        config_path: str = "configs/musetalk_v15.yaml",
        project_root: Optional[str] = None,
) -> str:
    cfg = load_yaml(config_path)

    model_path = cfg["models"]["s3fd"]["path"]

    model_path = resolve_path(
        model_path,
        project_root,
    )

    return model_path


class SFDDetector(FaceDetector):

    def __init__(
            self,
            device,
            path_to_detector: Optional[str] = None,
            verbose: bool = False,
            auto_download: bool = False,
    ):
        super().__init__(
            device,
            verbose,
        )

        self.device = str(device)

        if path_to_detector is None:
            raise ValueError(
                "path_to_detector is None. "
                "Please pass model path or use SFDDetector.from_config()."
            )

        if not os.path.isfile(path_to_detector):
            if auto_download:
                print(
                    f"[SFDDetector] model not found, download from: {models_urls['s3fd']}"
                )
                model_weights = load_url(
                    models_urls["s3fd"],
                    map_location="cpu",
                )
            else:
                raise FileNotFoundError(
                    f"S3FD detector weight not found: {path_to_detector}"
                )
        else:
            model_weights = torch.load(
                path_to_detector,
                map_location="cpu",
                weights_only=True,
            )

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(
            model_weights,
            strict=True,
        )

        self.face_detector.to(
            self.device,
        )

        self.face_detector.eval()

        print(
            f"[SFDDetector] load weight success: {path_to_detector}"
        )

    @classmethod
    def from_config(
            cls,
            config_path: str = "configs/musetalk_v15.yaml",
            project_root: Optional[str] = None,
            verbose: bool = False,
            auto_download: bool = False,
    ):
        device = get_device(
            config_path=config_path,
        )

        model_path = get_s3fd_path(
            config_path=config_path,
            project_root=project_root,
        )

        return cls(
            device=device,
            path_to_detector=model_path,
            verbose=verbose,
            auto_download=auto_download,
        )

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(
            tensor_or_path,
        )

        bboxlist = detect(
            self.face_detector,
            image,
            device=self.device,
        )

        keep = nms(
            bboxlist,
            0.3,
        )

        bboxlist = bboxlist[
            keep,
            :,
        ]

        bboxlist = [
            x for x in bboxlist
            if x[-1] > 0.5
        ]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(
            self.face_detector,
            images,
            device=self.device,
        )

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

        bboxlists = [
            [
                x for x in bboxlist
                if x[-1] > 0.5
            ]
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


if __name__ == "__main__":
    detector = SFDDetector.from_config(
        "configs/musetalk_v15.yaml",
    )

    img_path = "assets/sit.jpeg"

    bboxes = detector.detect_from_image(
        img_path,
    )

    print("bboxes:")
    for bbox in bboxes:
        print(bbox)