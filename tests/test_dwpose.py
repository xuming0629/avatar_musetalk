#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_dwpose.py
# @Time          : 2026-06-22 22:49:03
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

from src.nets.face.dwpose.dwpose import RTMPose

if __name__ == "__main__":
    img_path = "./assets/3456.png"

    rtmpose = RTMPose()

    save_path = rtmpose.predict_and_visualize(
        img_path=img_path,
        out_bbox=None,
        score_thr=0.3,
    )

    print(f"✅ 可视化结果已保存为: {save_path}")

    vis_img = cv2.imread(save_path)

    cv2.imshow(
        "Keypoints",
        vis_img,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()