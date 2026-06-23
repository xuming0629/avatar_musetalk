#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_unet.py
# @Time          : 2026-06-23 22:37:49
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : musetalk v1.5 unet 
# @Company       : 2026 XuMing. All Rights Reserved.
"""


import os
import sys
import cv2 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


from src.nets.musetalk.unet import UNet

if __name__ == "__main__":

    unet = UNet.from_config(
        "configs/musetalk_v15.yaml"
    )

    print("UNet load success")
    print("device:", unet.device)