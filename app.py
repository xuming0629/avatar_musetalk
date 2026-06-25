import os
import time
import pdb
import re

import gradio as gr
import numpy as np
import sys
import subprocess



def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)


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





def inference():
    pass




