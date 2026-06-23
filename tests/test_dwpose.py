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

    net = RTMPose.from_config(
        "configs/musetalk_v15.yaml"
    )

    save_path = net.predict_and_visualize(
        "assets/3456.png"
    )

    print(f"✅ 可视化结果已保存为: {save_path}")

    vis_img = cv2.imread(save_path)
    cv2.imshow("Keypoints", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()