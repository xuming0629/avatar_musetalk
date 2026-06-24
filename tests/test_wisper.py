#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_wisper.py
# @Time          : 2026-06-22 16:04:13
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   :
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""


import os
import sys
import cv2 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


from src.nets.whisper.audio2feature import Audio2Feature

if __name__ == "__main__":

    audio_processor = Audio2Feature.from_config(
        "configs/musetalk_v15.yaml"
    )

    audio_path = "./assets/test.mp3"

    array = audio_processor.audio2feat(
        audio_path
    )

    print(
        "audio feature:",
        array.shape
    )

    fps = 25

    chunks = audio_processor.feature2chunks(
        feature_array=array,
        fps=fps,
        audio_feat_length=[2, 2],
    )

    print(
        "chunks:",
        len(chunks),
        chunks[0].shape
    )