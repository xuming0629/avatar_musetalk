import os
import time
import pdb
import re

import gradio as gr
import numpy as np
import sys
import subprocess




# from src.utils.blending import get_image
# from src.utils.utils import get_file_type, get_video_fps, datagen
# from src.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range


from src.nets.face.parsing.FaceParsing import FaceParsing
from src.nets.face.dwpose.dwpose import RTMPose    # 人体关键点检测
from src.nets.face.parsing.face_parser import FaceParsing  # 
from src.nets.face.s3fd.sfd_detector import SFDDetector    # 人脸检测
from src.nets.musetalk.vae import VAE           
from src.nets.musetalk.unet import UNet, PositionalEncoding
from src.nets.face.alignment.face_alignment import FaceAlignment  # 人脸对其
from src.nets.whisper.audio2feature import Audio2Feature

## 先全局声明模型


####################################
vae = VAE()
unet = UNet()
pe = PositionalEncoding(d_model=384)
####################################

fa = FaceAlignment()
fp = FaceParsing()
dwpose = RTMPose()
sf_detect = SFDDetector()
####################################
# 还差 whisper 的类

print("==========================")


# ============================================================
# 1. 全局模型，只初始化一次
# ============================================================

print("========== Loading global models ==========")

vae = VAE()
unet = UNet()
pe = PositionalEncoding(d_model=384)

fa = FaceAlignment()
fp = FaceParsing()
dwpose = RTMPose()
sf_detect = SFDDetector()


# 还差 whisper / audio processor
# audio_processor = AudioProcessor()

print("========== Models loaded ==========")


# ============================================================
# 2. 默认参数构造函数
# ============================================================
from argparse import Namespace
def build_args(
    bbox_shift,
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
):
    """
    构造和 inference.py 对齐的参数。
    """
    args_dict = {
        "result_dir": "./results/output",
        "fps": 25,
        "batch_size": 8,
        "output_vid_name": "",
        "use_saved_coord": False,

        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,

        "version": "v15",

        "bbox_shift": bbox_shift,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width,
    }

    return Namespace(**args_dict)




#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import Namespace

import gradio as gr

from src.nets.musetalk.vae import VAE
from src.nets.musetalk.unet import UNet, PositionalEncoding
from src.nets.face.alignment.face_alignment import FaceAlignment
from src.nets.face.parsing.face_parser import FaceParsing
from src.nets.face.dwpose.dwpose import RTMPose
from src.nets.face.s3fd.sfd_detector import SFDDetector

# 后面你可以补 Whisper / AudioProcessor
# from src.utils.audio_processor import AudioProcessor


# ============================================================
# 1. 全局模型，只初始化一次
# ============================================================

print("========== Loading global models ==========")

vae = VAE()
unet = UNet()
pe = PositionalEncoding(d_model=384)

fa = FaceAlignment()
fp = FaceParsing()
dwpose = RTMPose()
sf_detect = SFDDetector()

# 还差 whisper / audio processor
# audio_processor = AudioProcessor()

print("========== Models loaded ==========")


# ============================================================
# 2. 默认参数构造函数
# ============================================================

def build_args(
    bbox_shift,
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
):
    """
    构造和 inference.py 对齐的参数。
    """
    args_dict = {
        "result_dir": "./results/output",
        "fps": 25,
        "batch_size": 8,
        "output_vid_name": "",
        "use_saved_coord": False,

        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,

        "version": "v15",

        "bbox_shift": bbox_shift,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width,
    }

    return Namespace(**args_dict)


# ============================================================
# 3. 推理函数
# ============================================================

def inference(
    audio_path,
    video_path,
    bbox_shift,
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Gradio 调用入口。

    注意：
    - 模型不在这里初始化
    - 这里只做参数接收、路径检查、调用 pipeline
    """

    if audio_path is None or not os.path.exists(audio_path):
        raise gr.Error("音频文件不存在")

    if video_path is None or not os.path.exists(video_path):
        raise gr.Error("视频/图片文件不存在")

    args = build_args(
        bbox_shift=bbox_shift,
        extra_margin=extra_margin,
        parsing_mode=parsing_mode,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
    )

    os.makedirs(args.result_dir, exist_ok=True)

    print("========== Inference args ==========")
    print(args)
    print("audio_path:", audio_path)
    print("video_path:", video_path)

    # ========================================================
    # 这里开始接你自己的 MuseTalk 推理流程
    # ========================================================
    #
    # 示例结构：
    #
    # output_path = run_musetalk_inference(
    #     audio_path=audio_path,
    #     video_path=video_path,
    #     args=args,
    #     vae=vae,
    #     unet=unet,
    #     pe=pe,
    #     fa=fa,
    #     fp=fp,
    #     dwpose=dwpose,
    #     sf_detect=sf_detect,
    #     audio_processor=audio_processor,
    #     progress=progress,
    # )
    #
    # return output_path
    #
    # ========================================================

    output_path = os.path.join(args.result_dir, "demo_output.mp4")

    # 临时占位，避免函数为空
    # 等你把真实 pipeline 接进来后，删除这一段
    if not os.path.exists(output_path):
        raise gr.Error(
            "当前 inference 框架已经整理好，但还没有接入真正的 MuseTalk 推理流程。"
        )

    return output_path



if __name__ == "__main__":
    



