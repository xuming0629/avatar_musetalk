#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_faceparsing.py
# @Time          : 2026-06-24 18:57:07
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TODO
# @Company       : 2026 XuMing. All Rights Reserved.
"""



import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2 
import numpy as np


from src.nets.face.parsing.face_parser import FaceParsing


if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('./assets/face_0.png')
    segmap.save('./assets/outputs/res.png')



