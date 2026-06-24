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