#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2 
import numpy as np


from src.nets.face.alignment.face_alignment import FaceAlignment


if __name__ == "__main__":
    import cv2

    face_alignment = FaceAlignment()

    img_path = "assets/sit.jpeg"
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"读取图片失败: {img_path}")

    bbox = face_alignment.detect_from_image(img)

    print("bbox:", bbox)

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        vis = img.copy()

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        save_path = "assets/sit_face_bbox.jpg"

        cv2.imwrite(
            save_path,
            vis,
        )

        print(f"save result to: {save_path}")