#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : vae.py
# @Time          : 2026-06-23 22:56:18
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : MuseTalk VAE wrapper with default config support
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import os
from typing import Any, Optional

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from diffusers import AutoencoderKL

from src.nets.common.config import (
    load_yaml,
    resolve_path,
    get_device,
)


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"


def get_vae_path_and_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
):
    """
    从配置文件读取 VAE 模型路径和参数。

    支持配置格式：

    models:
      sd-vae-ft-mse:
        path: models/sd-vae-ft-mse
        resized_img: 256
        use_float16: false

    或者：

    models:
      sd-vae:
        path: models/sd-vae
        resized_img: 256
        use_float16: false
    """

    cfg = load_yaml(config_path)

    models_cfg = cfg.get("models", {})

    # 优先兼容你当前代码里的 sd-vae-ft-mse
    model_cfg = models_cfg.get("sd-vae-ft-mse", None)

    # 再兼容 MuseTalk 官方常见 sd-vae
    if model_cfg is None:
        model_cfg = models_cfg.get("sd-vae", None)

    if model_cfg is None:
        raise ValueError(
            "Config error: models.sd-vae-ft-mse or models.sd-vae not found"
        )

    model_path = model_cfg.get("path", None)

    if model_path is None:
        raise ValueError(
            "Config error: VAE model path not found"
        )

    model_path = resolve_path(
        model_path,
        project_root,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"VAE model path not found: {model_path}"
        )

    resized_img = model_cfg.get(
        "resized_img",
        256,
    )

    use_float16 = model_cfg.get(
        "use_float16",
        False,
    )

    return {
        "model_path": model_path,
        "resized_img": resized_img,
        "use_float16": use_float16,
    }


class VAE:
    """
    MuseTalk VAE wrapper.

    支持默认直接声明：

        vae = VAE()

    等价于：

        vae = VAE.from_config("configs/musetalk_v15.yaml")

    也支持直接传模型路径：

        vae = VAE(model_path="models/sd-vae-ft-mse")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        resized_img: Optional[int] = None,
        device: Optional[str] = None,
        use_float16: Optional[bool] = None,
        auto_to_device: bool = True,
        eval_mode: bool = True,
    ):
        """
        参数:
            model_path:
                VAE 模型目录。如果不传，则自动从 config 读取。

            config_path:
                默认配置文件路径。

            project_root:
                项目根目录，用于 resolve_path。

            resized_img:
                VAE 输入尺寸。不传则从 config 读取，默认 256。

            device:
                cpu / cuda。不传则从 config 读取。

            use_float16:
                是否使用 fp16。不传则从 config 读取。

            auto_to_device:
                是否自动移动模型到 device。

            eval_mode:
                是否自动设置 eval。
        """

        self.config_path = config_path
        self.project_root = project_root

        if model_path is None:
            vae_cfg = get_vae_path_and_config(
                config_path=config_path,
                project_root=project_root,
            )

            model_path = vae_cfg["model_path"]

            if resized_img is None:
                resized_img = vae_cfg["resized_img"]

            if use_float16 is None:
                use_float16 = vae_cfg["use_float16"]

        else:
            model_path = resolve_path(
                model_path,
                project_root,
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"VAE model path not found: {model_path}"
                )

            if resized_img is None:
                resized_img = 256

            if use_float16 is None:
                use_float16 = False

        if device is None:
            device = get_device(
                config_path=config_path,
            )

        self.model_path = model_path
        self.device = torch.device(device)
        self._resized_img = int(resized_img)
        self._use_float16 = bool(use_float16)

        print(f"[VAE] load model from: {self.model_path}")

        self.vae = AutoencoderKL.from_pretrained(
            self.model_path,
        )

        if auto_to_device:
            self.vae.to(self.device)

        if self._use_float16:
            self.vae = self.vae.half()

        if eval_mode:
            self.vae.eval()

        self.scaling_factor = self.vae.config.scaling_factor

        self.transform = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        self._mask_tensor = self.get_mask_tensor().to(
            self.device,
        )

        print(
            f"[VAE] load success: {self.model_path}, "
            f"device={self.device}, "
            f"resized_img={self._resized_img}, "
            f"use_float16={self._use_float16}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
        resized_img: Optional[int] = None,
        use_float16: Optional[bool] = None,
        auto_to_device: bool = True,
        eval_mode: bool = True,
    ):
        """
        保留 from_config 写法，兼容旧代码。
        """

        return cls(
            model_path=None,
            config_path=config_path,
            project_root=project_root,
            resized_img=resized_img,
            device=device,
            use_float16=use_float16,
            auto_to_device=auto_to_device,
            eval_mode=eval_mode,
        )

    def get_mask_tensor(self) -> torch.Tensor:
        mask_tensor = torch.zeros(
            (
                self._resized_img,
                self._resized_img,
            ),
            dtype=torch.float32,
        )

        mask_tensor[: self._resized_img // 2, :] = 1.0
        mask_tensor[mask_tensor < 0.5] = 0.0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        return mask_tensor

    def preprocess_img(
        self,
        img: Any,
        half_mask: bool = False,
    ) -> torch.Tensor:
        """
        图片预处理。

        输入:
            img:
                - str: 图片路径
                - np.ndarray: OpenCV BGR 图片

        输出:
            torch.Tensor: [1, 3, H, W]
        """

        window = []

        if isinstance(img, str):
            image = cv2.imread(img)

            if image is None:
                raise FileNotFoundError(
                    f"读取图片失败: {img}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        elif isinstance(img, np.ndarray):
            image = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB,
            )

        else:
            raise TypeError(
                f"Unsupported image type: {type(img)}. "
                "Expected str or np.ndarray."
            )

        image = cv2.resize(
            image,
            (
                self._resized_img,
                self._resized_img,
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )

        window.append(image)

        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))

        x = torch.squeeze(
            torch.FloatTensor(x),
        )

        x = x.to(
            device=self.device,
            dtype=self.vae.dtype,
        )

        if half_mask:
            x = x * (
                self._mask_tensor > 0.5
            )

        x = self.transform(x)
        x = x.unsqueeze(0)

        return x

    def encode_latents(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        VAE encode。

        输入:
            image: [B, 3, H, W]

        输出:
            latents: [B, 4, H/8, W/8]
        """

        with torch.no_grad():
            init_latent_dist = self.vae.encode(
                image.to(
                    device=self.device,
                    dtype=self.vae.dtype,
                )
            ).latent_dist

            init_latents = (
                self.scaling_factor
                * init_latent_dist.sample()
            )

        return init_latents

    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> np.ndarray:
        """
        VAE decode。

        输入:
            latents: [B, 4, H, W]

        输出:
            image: BGR uint8 ndarray, [B, H, W, 3]
        """

        with torch.no_grad():
            latents = (
                1.0 / self.scaling_factor
            ) * latents

            image = self.vae.decode(
                latents.to(
                    device=self.device,
                    dtype=self.vae.dtype,
                )
            ).sample

        image = (
            image / 2 + 0.5
        ).clamp(0, 1)

        image = (
            image.detach()
                 .cpu()
                 .permute(0, 2, 3, 1)
                 .float()
                 .numpy()
        )

        image = (
            image * 255
        ).round().astype("uint8")

        # RGB -> BGR，方便 OpenCV 后处理
        image = image[..., ::-1]

        return image

    def get_latents_for_unet(
        self,
        img: Any,
    ) -> torch.Tensor:
        """
        获取 MuseTalk UNet 输入 latent。

        输出:
            latent_model_input: [1, 8, 32, 32]

        说明:
            前 4 通道：masked_latents
            后 4 通道：ref_latents
        """

        ref_image = self.preprocess_img(
            img,
            half_mask=True,
        )

        masked_latents = self.encode_latents(
            ref_image,
        )

        ref_image = self.preprocess_img(
            img,
            half_mask=False,
        )

        ref_latents = self.encode_latents(
            ref_image,
        )

        latent_model_input = torch.cat(
            [
                masked_latents,
                ref_latents,
            ],
            dim=1,
        )

        return latent_model_input

    def to(self, device):
        """
        手动切换 device。
        """

        self.device = torch.device(device)
        self.vae.to(self.device)
        self._mask_tensor = self._mask_tensor.to(self.device)

        return self

    def half(self):
        """
        手动切换 fp16。
        """

        self._use_float16 = True
        self.vae = self.vae.half()
        self._mask_tensor = self._mask_tensor.half()

        return self

    def float(self):
        """
        手动切换 fp32。
        """

        self._use_float16 = False
        self.vae = self.vae.float()
        self._mask_tensor = self._mask_tensor.float()

        return self

    @property
    def dtype(self):
        return self.vae.dtype


if __name__ == "__main__":
    vae = VAE()

    img_path = "./assets/3456.png"

    latents = vae.get_latents_for_unet(
        img_path,
    )

    print(
        "latents:",
        latents.shape,
        latents.dtype,
        latents.device,
    )