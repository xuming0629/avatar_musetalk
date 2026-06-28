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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_s3fd.py
# @Time          : 2026-06-22 15:41:20
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   : Test S3FD face detector
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""

import os
import sys
import argparse
import cv2
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.nets.face.s3fd.sfd_detector import SFDDetector


def clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox[:4]

    x1 = max(0, min(int(round(float(x1))), w - 1))
    y1 = max(0, min(int(round(float(y1))), h - 1))
    x2 = max(0, min(int(round(float(x2))), w))
    y2 = max(0, min(int(round(float(y2))), h))

    if x2 <= x1:
        x2 = min(w, x1 + 1)

    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def print_cuda_info():
    print("=" * 80)
    print("[Env] torch:", torch.__version__)
    print("[Env] cuda available:", torch.cuda.is_available())
    print("[Env] torch cuda:", torch.version.cuda)
    print("[Env] cudnn:", torch.backends.cudnn.version())
    print("[Env] cudnn enabled:", torch.backends.cudnn.enabled)
    print("[Env] cudnn benchmark:", torch.backends.cudnn.benchmark)
    print("[Env] gpu count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("[Env] current device:", torch.cuda.current_device())
        print("[Env] gpu name:", torch.cuda.get_device_name(0))

    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img",
        default="assets/3456.png",
        help="input image path",
    )

    parser.add_argument(
        "--out-dir",
        default="assets/outputs",
        help="output directory",
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=[None, "cpu", "cuda"],
        help="device: cpu or cuda. None means read from config.",
    )

    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="disable cudnn for debugging CUDA segmentation fault",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    img_path = args.img
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print_cuda_info()

    if args.disable_cudnn:
        print("[Debug] disable cudnn")
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w = img.shape[:2]

    print("[Image] path:", img_path)
    print("[Image] shape:", img.shape)

    if max(h, w) > 1280:
        print(
            "[Warning] image is large. "
            "If CUDA segfault happens, resize image before S3FD detection."
        )

    print("[Detector] init start")

    if args.device is None:
        detector = SFDDetector()
    else:
        detector = SFDDetector(device=args.device)

    print("[Detector] init done")
    print("[Detector] detect start")

    if str(detector.device).startswith("cuda"):
        torch.cuda.synchronize()
        print("[CUDA] before detect_from_image")
    
    for i in range(100):
        bboxes = detector.detect_from_image(img_path)

        if str(detector.device).startswith("cuda"):
            torch.cuda.synchronize()
            print("[CUDA] after detect_from_image")

        print("[Detector] detect done")

        print("bboxes:")
        for bbox in bboxes:
            print(bbox)

        if len(bboxes) == 0:
            print("No face detected")
            sys.exit(0)

        vis = img.copy()

        face_rois = []

        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = clip_bbox(bbox, w, h)

            print(f"[BBox] face_{idx}:", [x1, y1, x2, y2])

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

            if roi.size == 0:
                print(f"[Warning] empty roi, skip face_{idx}")
                continue

            face_rois.append((roi, [x1, y1, x2, y2]))

            roi_path = os.path.join(out_dir, f"face_roi_{idx}.jpg")
            cv2.imwrite(roi_path, roi)
            print("save roi:", roi_path)

        vis_path = os.path.join(out_dir, "s3fd_face_bbox.jpg")
        cv2.imwrite(vis_path, vis)
        print("save bbox image:", vis_path)

        if len(face_rois) == 0:
            print("No valid face roi")
            sys.exit(0)

        largest_idx = max(
            range(len(face_rois)),
            key=lambda i: (
                face_rois[i][1][2] - face_rois[i][1][0]
            )
            * (
                face_rois[i][1][3] - face_rois[i][1][1]
            ),
        )

        largest_roi = face_rois[largest_idx][0]
        largest_path = os.path.join(out_dir, "largest_face_roi.jpg")
        cv2.imwrite(largest_path, largest_roi)

        print("save largest roi:", largest_path)
