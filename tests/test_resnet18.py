#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_resnet18.py
# @Time          : 2026-06-22 14:03:16
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   :
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.nets.face.parsing.resnet import Resnet18
import torch




if __name__ == "__main__":

    net = Resnet18.from_config(
        "configs/musetalk_v15.yaml"
    )

    x = torch.randn(
        1,
        3,
        512,
        512
    )

    feat8, feat16, feat32 = net(x)

    print(feat8.shape)
    print(feat16.shape)
    print(feat32.shape)