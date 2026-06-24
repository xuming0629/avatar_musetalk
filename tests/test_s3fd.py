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
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.nets.face.s3fd.sfd_detector import SFDDetector


def clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox[:4]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


if __name__ == "__main__":
    img_path = "assets/test.png"
    out_dir = "assets/outputs"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w = img.shape[:2]

    detector = SFDDetector()

    bboxes = detector.detect_from_image(img_path)

    print("bboxes:")
    for bbox in bboxes:
        print(bbox)

    if len(bboxes) == 0:
        print("No face detected")
        sys.exit(0)

    vis = img.copy()

    # 保存所有检测框和所有 ROI
    face_rois = []

    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = clip_bbox(bbox, w, h)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            vis,
            f"face_{idx}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        roi = img[y1:y2, x1:x2].copy()
        face_rois.append((roi, [x1, y1, x2, y2]))

        roi_path = os.path.join(out_dir, f"face_roi_{idx}.jpg")
        cv2.imwrite(roi_path, roi)
        print("save roi:", roi_path)

    # 保存画框图
    vis_path = os.path.join(out_dir, "s3fd_face_bbox.jpg")
    cv2.imwrite(vis_path, vis)
    print("save bbox image:", vis_path)

    # 保存最大人脸
    largest_idx = max(
        range(len(face_rois)),
        key=lambda i: (face_rois[i][1][2] - face_rois[i][1][0])
        * (face_rois[i][1][3] - face_rois[i][1][1]),
    )

    largest_roi = face_rois[largest_idx][0]
    largest_path = os.path.join(out_dir, "largest_face_roi.jpg")
    cv2.imwrite(largest_path, largest_roi)

    print("save largest roi:", largest_path)
    
    
    
