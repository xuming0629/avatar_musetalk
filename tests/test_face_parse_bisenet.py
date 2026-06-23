#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_face_parse_bisenet.py
# @Time          : 2026-06-22 14:48:53
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   : Test for BiSeNet Face Parsing
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


from src.nets.face.parsing.face_parse_bisenet import BiSeNet
import torch

if __name__ == "__main__":

    net = BiSeNet.from_config(
        "configs/musetalk_v15.yaml"
    )

    net.eval()

    x = torch.randn(
        1,
        3,
        512,
        512
    )

    out, out16, out32 = net(x)

    print(out.shape)
    print(out16.shape)
    print(out32.shape)