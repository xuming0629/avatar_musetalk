#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_vae.py
# @Time          : 2026-06-23 22:21:27
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TODO
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import os
import sys
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.nets.musetalk.vae import VAE

if __name__ == "__main__":

    vae = VAE.from_config(
        "configs/musetalk_v15.yaml"
    )

    crop_imgs_path = "./results/sun001_crop/"
    latents_out_path = "./results/latents/"

    os.makedirs(
        latents_out_path,
        exist_ok=True
    )

    files = os.listdir(
        crop_imgs_path
    )

    files.sort()

    files = [
        file for file in files
        if file.lower().endswith(".png")
    ]

    for file in files:
        index = os.path.splitext(file)[0]

        img_path = os.path.join(
            crop_imgs_path,
            file
        )

        latents = vae.get_latents_for_unet(
            img_path
        )

        print(
            img_path,
            "latents",
            latents.size()
        )

        # torch.save(
        #     latents,
        #     os.path.join(
        #         latents_out_path,
        #         index + ".pt"
        #     )
        # )