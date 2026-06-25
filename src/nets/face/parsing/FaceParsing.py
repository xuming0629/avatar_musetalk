#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from .face_parse_bisenet import BiSeNet


class FaceParsing:
    def __init__(
        self,
        resnet_path="./models/resnet18/resnet18-5c106cde.pth",
        model_pth="./models/face-parse-bisent/79999_iter.pth",
        left_cheek_width=80,
        right_cheek_width=80,
    ):
        """
        Face parsing wrapper.

        Args:
            resnet_path: ResNet18 权重路径
            model_pth: BiSeNet face parsing 权重路径
            left_cheek_width: jaw 模式下左脸颊保护宽度
            right_cheek_width: jaw 模式下右脸颊保护宽度
        """

        self.left_cheek_width = int(left_cheek_width)
        self.right_cheek_width = int(right_cheek_width)

        # 自动检测设备
        self.device = self.get_device()
        print(f"[INFO] FaceParsing using device: {self.device}")

        # 初始化模型
        self.net = self.model_init(
            resnet_path=resnet_path,
            model_pth=model_pth,
        )

        self.preprocess = self.image_preprocess()

        # jaw 模式使用的下巴膨胀 kernel
        self.kernel = self._create_jaw_kernel()

        # cheek erosion kernel：压平脸颊区域，避免嘴部融合范围过大
        self.cheek_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (35, 3),
        )

        # 脸颊保护 mask
        self.cheek_mask = self._create_cheek_mask(
            left_cheek_width=self.left_cheek_width,
            right_cheek_width=self.right_cheek_width,
        )

    def get_device(self):
        """自动检测 CUDA / NPU / CPU。"""
        has_npu = False

        try:
            import torch_npu  # noqa: F401

            if hasattr(torch, "npu"):
                has_npu = torch.npu.is_available()
        except Exception:
            has_npu = False

        if torch.cuda.is_available():
            return torch.device("cuda")

        if has_npu:
            return torch.device("npu:0")

        return torch.device("cpu")

    def model_init(
        self,
        resnet_path,
        model_pth,
    ):
        """加载 BiSeNet 模型。"""

        if not os.path.exists(resnet_path):
            raise FileNotFoundError(
                f"ResNet18 权重不存在: {resnet_path}"
            )

        if not os.path.exists(model_pth):
            raise FileNotFoundError(
                f"FaceParsing 权重不存在: {model_pth}"
            )

        net = BiSeNet(resnet_path)
        net.to(self.device)

        map_location = self.device
        if self.device.type in ["cuda", "npu"]:
            map_location = None
        else:
            map_location = torch.device("cpu")

        state_dict = torch.load(
            model_pth,
            map_location=map_location,
        )

        # 兼容有些权重外面包了一层 state_dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # 兼容 DataParallel 的 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            new_state_dict[k] = v

        net.load_state_dict(
            new_state_dict,
            strict=True,
        )

        net.eval()
        return net

    def image_preprocess(self):
        """标准化预处理。"""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def _create_jaw_kernel(self):
        """
        创建 jaw 模式下的下巴延展 kernel。

        形状类似：
            上半部分为空
            下半部分逐渐变宽
            底部继续向下延伸
        """
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height

        kernel = np.zeros(
            (total_size, total_size),
            dtype=np.uint8,
        )

        center_x = total_size // 2

        # cone part
        for row in range(cone_height):
            if row < cone_height // 2:
                continue

            width = int(
                2 * (row - cone_height // 2) + 1
            )

            start = int(
                center_x - width // 2
            )

            end = int(
                center_x + width // 2 + 1
            )

            kernel[row, start:end] = 1

        if cone_height > 0:
            base_width = int(
                kernel[cone_height - 1].sum()
            )
        else:
            base_width = 1

        # vertical extension part
        for row in range(cone_height, total_size):
            start = max(
                0,
                int(center_x - base_width // 2),
            )

            end = min(
                total_size,
                int(center_x + base_width // 2 + 1),
            )

            kernel[row, start:end] = 1

        return kernel

    def _create_cheek_mask(
        self,
        left_cheek_width=80,
        right_cheek_width=80,
    ):
        """
        创建脸颊区域 mask。

        mask 中为 255 的区域表示左右脸颊区域。
        jaw 模式下会对这部分做额外约束，避免脸颊被过度融合。
        """
        left_cheek_width = int(left_cheek_width)
        right_cheek_width = int(right_cheek_width)

        mask = np.zeros(
            (512, 512),
            dtype=np.uint8,
        )

        center = 512 // 2

        # 左脸颊区域
        left_end = max(
            0,
            center - left_cheek_width,
        )

        cv2.rectangle(
            mask,
            (0, 0),
            (left_end, 512),
            255,
            -1,
        )

        # 右脸颊区域
        right_start = min(
            512,
            center + right_cheek_width,
        )

        cv2.rectangle(
            mask,
            (right_start, 0),
            (512, 512),
            255,
            -1,
        )

        return mask

    def _prepare_image(
        self,
        image,
    ):
        """
        输入统一转成 PIL RGB。
        支持：
            - str 路径
            - PIL.Image
            - np.ndarray BGR/RGB
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            return image

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            # 默认认为 OpenCV 读入的是 BGR
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

            image = Image.fromarray(
                image.astype(np.uint8)
            ).convert("RGB")

            return image

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    def _to_device(
        self,
        tensor,
    ):
        if self.device.type == "npu":
            return tensor.to(self.device)
        return tensor.to(self.device)

    def __call__(
        self,
        image,
        size=(512, 512),
        mode="raw",
    ):
        """
        执行分割预测。

        Args:
            image: 图片路径 / PIL.Image / np.ndarray
            size: 模型输入尺寸，默认 512x512
            mode:
                raw:
                    face + lips，原始脸部融合区域
                jaw:
                    face + lips + 下巴区域，适合 MuseTalk 嘴部融合
                neck:
                    face + lips + neck

        Returns:
            PIL.Image，单通道 mask，0/255
        """

        image = self._prepare_image(image)

        with torch.no_grad():
            image = image.resize(
                size,
                Image.BILINEAR,
            )

            img = self.preprocess(image)
            img = torch.unsqueeze(
                img,
                0,
            )

            img = self._to_device(img)

            out = self.net(img)[0]

            parsing = (
                out.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .argmax(0)
            )

            # ------------------------------------------------
            # mode 逻辑
            # ------------------------------------------------
            if mode == "neck":
                # 1: skin
                # 11,12,13: lips
                # 14: neck
                parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
                parsing[parsing != 255] = 0

            elif mode == "jaw":
                # 只取 face class 作为基础区域
                face_region = np.isin(
                    parsing,
                    [1],
                ).astype(np.uint8) * 255

                # 下巴区域向下扩张
                original_dilated = cv2.dilate(
                    face_region,
                    self.kernel,
                    iterations=1,
                )

                # 对左右脸颊做扁平腐蚀，避免脸颊融合范围太宽
                eroded = cv2.erode(
                    original_dilated,
                    self.cheek_kernel,
                    iterations=2,
                )

                # 脸颊区域用 eroded，中间区域保留 original_dilated
                cheek_part = cv2.bitwise_and(
                    eroded,
                    self.cheek_mask,
                )

                middle_part = cv2.bitwise_and(
                    original_dilated,
                    cv2.bitwise_not(self.cheek_mask),
                )

                face_region = cv2.bitwise_or(
                    cheek_part,
                    middle_part,
                )

                # 排除 nose=10，避免鼻子被过度融合
                parsing[
                    (face_region == 255)
                    & (~np.isin(parsing, [10]))
                ] = 255

                # 加上嘴唇
                parsing[np.isin(parsing, [11, 12, 13])] = 255

                parsing[parsing != 255] = 0

            else:
                # raw 默认：
                # 1: skin
                # 11,12,13: lips
                parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
                parsing[parsing != 255] = 0

        parsing = Image.fromarray(
            parsing.astype(np.uint8)
        )

        return parsing


