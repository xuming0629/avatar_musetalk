#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸检测器基类。

说明：
- 保持原接口不变；
- 支持图片路径、numpy.ndarray、torch.Tensor 输入；
- SFDDetector 内部会显式传 rgb=False，让 S3FD 使用 BGR 输入。
"""

import glob
import logging

import cv2
import numpy as np
import torch
from tqdm import tqdm


class FaceDetector(object):
    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose

        device_str = str(device)

        if verbose and "cpu" in device_str:
            logger = logging.getLogger(__name__)
            logger.warning("Detection running on CPU, this may be potentially slow.")

        if "cpu" not in device_str and "cuda" not in device_str:
            if verbose:
                logger = logging.getLogger(__name__)
                logger.error(
                    "Expected values for device are: {cpu, cuda} but got: %s",
                    device_str,
                )
            raise ValueError

    def detect_from_image(self, tensor_or_path):
        raise NotImplementedError

    def detect_from_directory(
        self,
        path,
        extensions=[".jpg", ".png"],
        recursive=False,
        show_progress_bar=True,
    ):
        """
        对目录内图片进行批量人脸检测。
        """
        logger = logging.getLogger(__name__) if self.verbose else None

        if len(extensions) == 0:
            if self.verbose:
                logger.error("Expected at least one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")

        additional_pattern = "/**/*" if recursive else "/*"

        files = []
        for extension in extensions:
            files.extend(
                glob.glob(
                    path + additional_pattern + extension,
                    recursive=recursive,
                )
            )

        if self.verbose:
            logger.info("Finished searching for images. %s images found", len(files))
            logger.info("Preparing to run the detection.")

        predictions = {}

        for image_path in tqdm(files, disable=not show_progress_bar):
            if self.verbose:
                logger.info("Running the face detector on image: %s", image_path)

            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info("The detector was successfully run on all %s images", len(files))

        return predictions

    @property
    def reference_scale(self):
        raise NotImplementedError

    @property
    def reference_x_shift(self):
        raise NotImplementedError

    @property
    def reference_y_shift(self):
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """
        将输入统一转成 numpy.ndarray。

        Args:
            tensor_or_path:
                - str: 图片路径；
                - np.ndarray: 图片数组；
                - torch.Tensor: 图片 Tensor。
            rgb:
                - True: 返回 RGB；
                - False: 返回 BGR。

        注意：
            cv2.imread 默认读出来是 BGR。
            S3FD 使用 BGR 均值 [104, 117, 123]，所以 SFDDetector 会传 rgb=False。
        """
        if isinstance(tensor_or_path, str):
            img = cv2.imread(tensor_or_path)

            if img is None:
                raise FileNotFoundError(f"Image not found or unreadable: {tensor_or_path}")

            return img[..., ::-1].copy() if rgb else img

        if torch.is_tensor(tensor_or_path):
            arr = tensor_or_path.detach().cpu().numpy()
            return arr if rgb else arr[..., ::-1].copy()

        if isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path if rgb else tensor_or_path[..., ::-1].copy()

        raise TypeError(f"Unsupported input type: {type(tensor_or_path)}")
