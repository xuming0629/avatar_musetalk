#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : vae.py
# @Time          : 2026-06-23 22:56:18
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TODO
# @Company       : 2026 XuMing. All Rights Reserved.
"""




import os
import sys


from typing import Any

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


class VAE:
    """
    VAE wrapper.
    """

    def __init__(
        self,
        model_path: str = "models/sd-vae-ft-mse",
        resized_img: int = 256,
        device: str = "cpu",
        use_float16: bool = False,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self._resized_img = int(resized_img)
        self._use_float16 = bool(use_float16)

        self.vae = AutoencoderKL.from_pretrained(
            self.model_path
        )

        self.vae.to(self.device)
        self.vae.eval()

        if self._use_float16:
            self.vae = self.vae.half()

        self.scaling_factor = self.vae.config.scaling_factor

        self.transform = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        self._mask_tensor = self.get_mask_tensor().to(
            self.device
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/musetalk_v15.yaml",
    ):
        cfg = load_yaml(config_path)

        model_cfg = (
            cfg.get("models", {})
               .get("sd-vae-ft-mse", {})
        )

        model_path = model_cfg.get("path")

        if model_path is None:
            raise ValueError(
                "Config error: models.sd-vae-ft-mse.path not found"
            )

        resized_img = model_cfg.get(
            "resized_img",
            256
        )

        use_float16 = model_cfg.get(
            "use_float16",
            False
        )

        return cls(
            model_path=model_path,
            resized_img=resized_img,
            device=get_device(config_path),
            use_float16=use_float16,
        )

    def get_mask_tensor(self) -> torch.Tensor:
        mask_tensor = torch.zeros(
            (
                self._resized_img,
                self._resized_img
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
        window = []

        if isinstance(img, str):
            image = cv2.imread(img)

            if image is None:
                raise FileNotFoundError(
                    f"读取图片失败: {img}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        else:
            image = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

        image = cv2.resize(
            image,
            (
                self._resized_img,
                self._resized_img
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )

        window.append(image)

        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))
        x = torch.squeeze(
            torch.FloatTensor(x)
        )

        x = x.to(self.device)

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

        image = image[..., ::-1]

        return image

    def get_latents_for_unet(
        self,
        img: Any,
    ) -> torch.Tensor:
        ref_image = self.preprocess_img(
            img,
            half_mask=True,
        )

        masked_latents = self.encode_latents(
            ref_image
        )

        ref_image = self.preprocess_img(
            img,
            half_mask=False,
        )

        ref_latents = self.encode_latents(
            ref_image
        )

        latent_model_input = torch.cat(
            [
                masked_latents,
                ref_latents,
            ],
            dim=1,
        )

        return latent_model_input


# if __name__ == "__main__":

#     vae = VAE.from_config(
#         "configs/musetalk_v15.yaml"
#     )

#     crop_imgs_path = "./results/sun001_crop/"
#     latents_out_path = "./results/latents/"

#     os.makedirs(
#         latents_out_path,
#         exist_ok=True
#     )

#     files = os.listdir(
#         crop_imgs_path
#     )

#     files.sort()

#     files = [
#         file for file in files
#         if file.lower().endswith(".png")
#     ]

#     for file in files:
#         index = os.path.splitext(file)[0]

#         img_path = os.path.join(
#             crop_imgs_path,
#             file
#         )

#         latents = vae.get_latents_for_unet(
#             img_path
#         )

#         print(
#             img_path,
#             "latents",
#             latents.size()
#         )

#         # torch.save(
#         #     latents,
#         #     os.path.join(
#         #         latents_out_path,
#         #         index + ".pt"
#         #     )
#         # )