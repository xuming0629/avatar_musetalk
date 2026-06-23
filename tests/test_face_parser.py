
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_face_parser.py
# @Time          : 2026-06-22 15:08:58
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   : Test for Face Parser
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""


import os
import sys
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.nets.face.parsing.face_parser import FaceParsing
# from nets.face.parsing.FaceParsing import FaceParsing





# if __name__ == "__main__":
#     parser = FaceParser(
#         config_path="configs/musetalk_v15.yaml",
#     )

#     save_path = parser.save(
#         image="./assets/face_0.png",
#         save_path="./assets/outputs/res.png",
#         mode="raw",
#     )

#     print(
#         f"[FaceParser] save mask: {save_path}"
#     )


# if __name__ == "__main__":
#     fp = FaceParsing()
#     segmap = fp('./assets/face_0.png')
#     segmap.save('./assets/outputs/res.png')

if __name__ == "__main__":
    fp = FaceParsing(
        config_path="configs/musetalk_v15.yaml",
    )

    save_path = fp.save(
        image="./assets/face_0.png",
        save_path="./assets/outputs/res.png"
    )

    print(f"[FaceParsing] save: {save_path}")