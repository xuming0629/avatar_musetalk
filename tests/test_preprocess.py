#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_s3fd.py
# @Time          : 2026-06-22 15:41:20
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   :
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.preprocessing import LandmarkBBoxExtractor
import pickle


if __name__ == "__main__":

    extractor = LandmarkBBoxExtractor.from_config(
        "configs/musetalk_v15.yaml"
    )

    img_list = [
        "./assets/imgs/test.png"
    ]
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    crop_coord_path = "./coord_face.pkl"

    coords_list, full_frames = extractor.get_landmark_and_bbox(
        img_list
    )

    with open(
        crop_coord_path,
        "wb",
    ) as f:
        pickle.dump(
            coords_list,
            f,
        )

    for bbox, frame in zip(
        coords_list,
        full_frames,
    ):
        if bbox == coord_placeholder:
            continue

        x1, y1, x2, y2 = bbox

        crop_frame = frame[
            y1:y2,
            x1:x2,
        ]

        print(
            "Cropped shape",
            crop_frame.shape,
        )

    print(
        coords_list
    )