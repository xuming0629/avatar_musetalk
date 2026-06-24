#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : simple_musetalk.py
# @Time          : 2026-06-24 13:59:34
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : 简单的数字人
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import os
import glob
import shutil
from pathlib import Path
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


####################################
# 底层基础模型类
from src.nets.face.dwpose.dwpose import RTMPose    # 人体关键点检测
from src.nets.face.parsing.face_parser import FaceParsing  # 
from src.nets.face.s3fd.sfd_detector import SFDDetector    # 人脸检测
from src.nets.musetalk.vae import VAE           
from src.nets.musetalk.unet import UNet 
from src.nets.face.alignment.face_alignment import FaceAlignment  # 人脸对其
from src.nets.common.config import get_device
####################################




def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def is_video_file(file_path):
    video_exts = [".mp4", ".mkv", ".flv", ".avi", ".mov"]
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


class MusetalkAvatarBuilder:
    def __init__(
        self,
        config_path: str,
        output_root: str = "outputs/avatars",
    ):
        self.config_path = config_path
        self.output_root = output_root

        self.device = get_device(config_path)

        self.rtmpose = RTMPose.from_config(config_path)

        self.face_detector = SFDDetector.from_config(config_path)
        # 
        # FaceAlignment.from_config(config_path)

        self.vae = VAE.from_config(config_path)

        self.face_parsing = FaceParsing.from_config(config_path)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        print("current_dir:", current_dir)
        print("parent_dir:", parent_dir)
        
    def face_seg(self, image):
        seg_image = self.face_parsing(image)
        if seg_image is None:
            print("error, no person_segment")
            return None

        seg_image = seg_image.resize(image.size)
        return seg_image
    
    
    def get_landmark_and_bbox(
        self,
        img_list,
        upperbondrange=0,
        avatar_id=None,
    ):
        """
        获取人脸关键点和 bbox。

        参数:
            img_list: 图片路径列表
            upperbondrange: bbox 上边界手动偏移
                > 0 向下
                < 0 向上
                = 0 使用默认
            avatar_id: 可选，用于进度更新

        返回:
            coords_list: 每帧人脸 bbox 列表
            frames: 读取后的图片帧列表
        """

        frames = read_imgs(img_list)


        batch_size_fa = 1
        batches = [
            frames[i:i + batch_size_fa]
            for i in range(0, len(frames), batch_size_fa)
        ]

        coords_list = []
        landmarks = []

        if upperbondrange != 0:
            print(
                "get key_landmark and face bounding boxes "
                f"with the bbox_shift: {upperbondrange}"
            )
        else:
            print("get key_landmark and face bounding boxes with the default value")

        average_range_minus = []
        average_range_plus = []

        coord_placeholder = (0.0, 0.0, 0.0, 0.0)

        for fb in tqdm(batches):
            frame_np = np.asarray(fb)

            # RTMPose 关键点检测
            keypoints, scores = self.rtmpose.predict(frame_np[0])

            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)

            landmarks.append(face_land_mark)

            # 人脸检测 bbox
            bbox = self.face_detector.detect_from_batch(frame_np)

            for j, f in enumerate(bbox):
                if f is None:
                    coords_list.append(coord_placeholder)
                    continue

                half_face_coord = face_land_mark[29].copy()

                range_minus = int((face_land_mark[30] - face_land_mark[29])[1])
                range_plus = int((face_land_mark[29] - face_land_mark[28])[1])

                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)

                if upperbondrange != 0:
                    # + 向下，- 向上
                    half_face_coord[1] = upperbondrange + half_face_coord[1]

                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                upper_bond = half_face_coord[1] - half_face_dist

                f_landmark = (
                    int(np.min(face_land_mark[:, 0])),
                    int(upper_bond),
                    int(np.max(face_land_mark[:, 0])),
                    int(np.max(face_land_mark[:, 1])),
                )

                x1, y1, x2, y2 = f_landmark

                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                    coords_list.append(f)

                    w = f[2] - f[0]
                    h = f[3] - f[1]

                    print("error bbox:", f)
                    print("fallback bbox size:", w, h)
                else:
                    coords_list.append(f_landmark)
        return coords_list, frames
    
    def get_image_prepare_material(self, image, face_box, upper_boundary_ratio=0.5, expand=1.2):
        body = Image.fromarray(image[:, :, ::-1])

        x, y, x1, y1 = face_box
        # print(x1-x,y1-y)
        crop_box, s = get_crop_box(face_box, expand)
        x_s, y_s, x_e, y_e = crop_box

        face_large = body.crop(crop_box)
        ori_shape = face_large.size

        mask_image = self.face_seg(face_large)
        mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        mask_image = Image.new('L', ori_shape, 0)
        mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

        # keep upper_boundary_ratio of talking area
        width, height = mask_image.size
        top_boundary = int(height * upper_boundary_ratio)
        modified_mask_image = Image.new('L', ori_shape, 0)
        modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

        blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
        mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
        return mask_array, crop_box

    def create_musetalk_huma(
        self,
        file: str,
        avatar_id: str = "default_avatar",
    ):
        """
        创建 MuseTalk 数字人素材。

        支持：
        1. 输入视频文件：自动抽帧
        2. 输入单张图片：复制到目标目录
        3. 输入图片文件夹：复制 png/jpg/jpeg 图片
        """

        save_full_path = os.path.join(
            self.output_root,
            avatar_id,
            "full_imgs",
        )

        os.makedirs(save_full_path, exist_ok=True)

        if os.path.isfile(file):
            if is_video_file(file):
                print(f"input is video: {file}")
                video2imgs(
                    file,
                    save_full_path,
                    ext="png",
                )
            else:
                print(f"input is image: {file}")
                dst_path = os.path.join(
                    save_full_path,
                    os.path.basename(file),
                )
                shutil.copyfile(file, dst_path)

        elif os.path.isdir(file):
            print(f"input is image directory: {file}")

            files = os.listdir(file)
            files.sort()

            img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

            for filename in files:
                src_path = os.path.join(file, filename)

                if not os.path.isfile(src_path):
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext not in img_exts:
                    continue

                dst_path = os.path.join(save_full_path, filename)
                shutil.copyfile(src_path, dst_path)

        else:
            raise FileNotFoundError(f"输入文件或目录不存在: {file}")

        input_img_list = sorted(
            glob.glob(os.path.join(save_full_path, "*.[jpJP][pnPN]*[gG]"))
        )

        if len(input_img_list) == 0:
            raise RuntimeError(f"没有找到有效图片: {save_full_path}")

        print(f"共找到 {len(input_img_list)} 张图片")
        print("extracting landmarks...")
        coord_list, frame_list = self.get_landmark_and_bbox(input_img_list, 5, avatar_id)
        print(coord_list, frame_list)
        
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

            print(latents)
            
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        mask_coords_list_cycle = []
        mask_list_cycle = []
        for i, frame in enumerate(tqdm(frame_list_cycle)):
            cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)
            face_box = coord_list_cycle[i]
            mask, crop_box = self.get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
            mask_coords_list_cycle += [crop_box]
            mask_list_cycle.append(mask)
        # common_util.update_avatar_progress(avatar_id, 100)
        with open(mask_coords_path, 'wb') as f:
            pickle.dump(mask_coords_list_cycle, f)

        with open(coords_path, 'wb') as f:
            pickle.dump(coord_list_cycle, f)
        torch.save(input_latent_list_cycle, os.path.join(latents_out_path))
        
        
        
    
        
        
if __name__ == "__main__":
    
    builder = MusetalkAvatarBuilder(
        config_path="configs/musetalk_v15.yaml",
        output_root="outputs/avatars",
    )

    result = builder.create_musetalk_huma(
        file="assets/imgs/3456.png",
        avatar_id="avatar_001",
    )
    
    


    