#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.nets.common.config import load_yaml, resolve_path, get_device
from src.nets.face.parsing.face_parse_bisenet import BiSeNet


class FaceParsing:
    def __init__(
        self,
        config_path="configs/musetalk_v15.yaml",
        project_root=None,
        left_cheek_width=80,
        right_cheek_width=80,
    ):
        self.config_path = config_path
        self.project_root = project_root

        self.left_cheek_width = int(left_cheek_width)
        self.right_cheek_width = int(right_cheek_width)

        self.cfg = load_yaml(config_path)
        self.device = get_device(config_path)

        print(f"[FaceParsing] device: {self.device}")

        self.resnet_path = self._get_resnet_path()
        self.model_path = self._get_face_parse_path()
        self.num_classes = self._get_num_classes()

        self.net = self.model_init()
        self.preprocess = self.image_preprocess()

        # jaw 模式下巴扩张 kernel
        self.kernel = self._create_jaw_kernel()

        # 脸颊区域压缩 kernel，避免左右脸颊融合区域过大
        self.cheek_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (35, 3),
        )

        # 脸颊保护 mask
        self.cheek_mask = self._create_cheek_mask(
            left_cheek_width=self.left_cheek_width,
            right_cheek_width=self.right_cheek_width,
        )

    def _get_resnet_path(self):
        path = self.cfg["models"]["resnet18"]["path"]
        path = resolve_path(path, self.project_root)
        return path

    def _get_face_parse_path(self):
        path = self.cfg["models"]["face-parse-bisent"]["path"]
        path = resolve_path(path, self.project_root)
        return path

    def _get_num_classes(self):
        return self.cfg["models"]["face-parse-bisent"].get(
            "num_classes",
            19,
        )

    def model_init(self):
        if not os.path.exists(self.resnet_path):
            raise FileNotFoundError(
                f"[FaceParsing] ResNet18 权重不存在: {self.resnet_path}"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[FaceParsing] BiSeNet 权重不存在: {self.model_path}"
            )

        net = BiSeNet(
            resnet_path=self.resnet_path,
            n_classes=self.num_classes,
        )

        try:
            state_dict = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # 兼容低版本 torch，没有 weights_only 参数
            state_dict = torch.load(
                self.model_path,
                map_location="cpu",
            )

        # 兼容 {'state_dict': xxx} 格式
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # 兼容 DataParallel 的 module. 前缀
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            clean_state_dict[k] = v

        net.load_state_dict(
            clean_state_dict,
            strict=False,
        )

        net.to(self.device)
        net.eval()

        print(f"[FaceParsing] load BiSeNet: {self.model_path}")
        print(f"[FaceParsing] load ResNet18: {self.resnet_path}")
        print(f"[FaceParsing] num_classes: {self.num_classes}")

        return net

    def image_preprocess(self):
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
        创建 jaw 模式的下巴扩张 kernel。

        这个 kernel 形状类似一个向下的锥形 + 竖直尾巴，
        用于让 face mask 往下巴区域扩张。
        """
        cone_height = 21
        tail_height = 12
        total_size = cone_height + tail_height

        kernel = np.zeros(
            (total_size, total_size),
            dtype=np.uint8,
        )

        center_x = total_size // 2

        # 锥形区域
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

        # 底部竖直延展区域
        if cone_height > 0:
            base_width = int(
                kernel[cone_height - 1].sum()
            )
        else:
            base_width = 1

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
        创建左右脸颊 mask。

        mask=255 的位置是左右脸颊区域。
        jaw 模式下，这部分区域会用 erode 后的 mask，
        防止脸颊区域融合太宽。
        """
        left_cheek_width = int(left_cheek_width)
        right_cheek_width = int(right_cheek_width)

        mask = np.zeros(
            (512, 512),
            dtype=np.uint8,
        )

        center = 512 // 2

        # 左脸颊
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

        # 右脸颊
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

    def _prepare_image(self, image):
        """
        统一输入格式为 PIL RGB。

        支持：
            - str 图片路径
            - np.ndarray
            - PIL.Image
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2RGB,
                )
            elif image.ndim == 3 and image.shape[2] == 3:
                # 注意：这里按 OpenCV BGR 输入处理
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2RGB,
                )
            else:
                raise ValueError(
                    f"Unsupported ndarray shape: {image.shape}"
                )

            image = Image.fromarray(
                image.astype(np.uint8)
            ).convert("RGB")

        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        return image

    def _build_binary_mask(
        self,
        parsing,
        mode="raw",
    ):
        """
        根据 parsing 类别图构造 0/255 mask。

        常见 face parsing 类别：
            1: skin
            10: nose
            11/12/13: lips / mouth
            14: neck
        """
        mode = str(mode).lower()

        if mode == "neck":
            # face + lips + neck
            parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
            parsing[parsing != 255] = 0
            return parsing

        if mode == "jaw":
            # 基础脸部区域
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

            # 左右脸颊做压缩，防止融合太宽
            eroded = cv2.erode(
                original_dilated,
                self.cheek_kernel,
                iterations=2,
            )

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

            # 排除鼻子 10
            parsing[
                (face_region == 255)
                & (~np.isin(parsing, [10]))
            ] = 255

            # 加嘴唇
            parsing[np.isin(parsing, [11, 12, 13])] = 255

            parsing[parsing != 255] = 0
            return parsing

        # raw 默认：face + lips
        parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
        parsing[parsing != 255] = 0
        return parsing

    def __call__(
        self,
        image,
        size=(512, 512),
        mode="raw",
    ):
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
            ).to(self.device)

            out = self.net(img)[0]

            parsing = (
                out.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .argmax(0)
                .astype(np.uint8)
            )

            parsing = self._build_binary_mask(
                parsing=parsing,
                mode=mode,
            )

        parsing = Image.fromarray(
            parsing.astype(np.uint8)
        )

        return parsing

    def save(
        self,
        image,
        save_path,
        size=(512, 512),
        mode="raw",
    ):
        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True,
        )

        mask = self(
            image=image,
            size=size,
            mode=mode,
        )

        mask.save(save_path)
        return save_path

