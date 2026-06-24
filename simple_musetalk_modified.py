#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : simple_musetalk.py
# @Time          : 2026-06-24 13:59:34
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : 简单的数字人素材构建脚本
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import glob
import os
import pickle
import shutil
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

####################################
# 底层基础模型类
from src.nets.face.dwpose.dwpose import RTMPose  # 人体关键点检测
from src.nets.face.parsing.face_parser import FaceParsing
from src.nets.face.s3fd.sfd_detector import SFDDetector  # 人脸检测
from src.nets.musetalk.vae import VAE
from src.nets.face.alignment.face_alignment import FaceAlignment  # 人脸对齐/检测
from src.nets.common.config import get_device
from src.nets.whisper.audio2feature import Audio2Feature
from src.nets.musetalk.unet import UNet
####################################


Coord = Tuple[int, int, int, int]
COORD_PLACEHOLDER: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mkv", ".flv", ".avi", ".mov")


def is_video_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTS


def read_imgs(img_list: Sequence[str]) -> List[np.ndarray]:
    frames = []
    print("reading images...")

    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue
        frames.append(frame)

    return frames


def video2imgs(
    vid_path: str,
    save_path: str,
    ext: str = "png",
    cut_frame: int = 10000000,
) -> int:
    """视频抽帧到 save_path。返回抽帧数量。"""

    os.makedirs(save_path, exist_ok=True)
    ext = ext.lstrip(".")

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {vid_path}")

    count = 0
    while True:
        if count >= cut_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(save_path, f"{count:08d}.{ext}")
        cv2.imwrite(out_path, frame)
        count += 1

    cap.release()
    print(f"video frames saved: {count}, dir: {save_path}")
    return count


def list_images(img_dir: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def get_crop_box(box, expand: float):
    x, y, x1, y1 = [int(round(v)) for v in box]
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def safe_clip_bbox(bbox, image_shape) -> Optional[Coord]:
    """把 bbox 限制到图像内部，非法返回 None。"""

    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def safe_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    bbox = safe_clip_bbox(bbox, frame.shape)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    return crop


class MusetalkAvatarBuilder:
    def __init__(
        self,
        config_path: str,
        output_root: str = "outputs/avatars",
        use_face_alignment: bool = False,
    ):
        self.config_path = config_path
        self.output_root = output_root

        self.device = get_device(config_path)
        self.rtmpose = RTMPose.from_config(config_path)

        # 默认保留你当前的 SFDDetector；如果 SFDDetector 没有 detect_from_batch，
        # detect_face_bbox() 会自动尝试 detect()。
        if use_face_alignment:
            self.face_detector = FaceAlignment.from_config(config_path)
        else:
            self.face_detector = SFDDetector.from_config(config_path)

        self.vae = VAE.from_config(config_path)
        self.face_parsing = FaceParsing(config_path)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        print("current_dir:", current_dir)
        print("parent_dir:", parent_dir)

    def get_avatar_paths(self, avatar_id: str):
        avatar_path = os.path.join(self.output_root, avatar_id)
        full_imgs_path = os.path.join(avatar_path, "full_imgs")
        mask_out_path = os.path.join(avatar_path, "mask")
        coords_path = os.path.join(avatar_path, "coords.pkl")
        mask_coords_path = os.path.join(avatar_path, "mask_coords.pkl")
        latents_out_path = os.path.join(avatar_path, "latents.pt")

        os.makedirs(avatar_path, exist_ok=True)
        os.makedirs(full_imgs_path, exist_ok=True)
        os.makedirs(mask_out_path, exist_ok=True)

        return {
            "avatar_path": avatar_path,
            "full_imgs_path": full_imgs_path,
            "mask_out_path": mask_out_path,
            "coords_path": coords_path,
            "mask_coords_path": mask_coords_path,
            "latents_out_path": latents_out_path,
        }

    def face_seg(self, image: Image.Image) -> Optional[Image.Image]:
        seg_image = self.face_parsing(image)
        if seg_image is None:
            print("[WARN] face parsing failed")
            return None

        seg_image = seg_image.resize(image.size)
        return seg_image

    def detect_face_bbox(self, frame_np: np.ndarray):
        """
        兼容不同 face detector 接口：
        1. detect_from_batch(frame_np)
        2. detect(frame)
        """

        if hasattr(self.face_detector, "detect_from_batch"):
            return self.face_detector.detect_from_batch(frame_np)

        if hasattr(self.face_detector, "detect"):
            bboxes = []
            for frame in frame_np:
                det = self.face_detector.detect(frame)
                bboxes.append(det)
            return bboxes

        raise AttributeError(
            "face_detector does not have detect_from_batch() or detect(). "
            "Please use FaceAlignment or check SFDDetector API."
        )

    def normalize_detector_bbox(self, f) -> Optional[Tuple[float, float, float, float]]:
        """兼容 detector 返回 [x1,y1,x2,y2] 或 [[x1,y1,x2,y2,score], ...]。"""

        if f is None:
            return None

        arr = np.asarray(f)
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            # 多个人脸时取面积最大的框
            boxes = arr[:, :4]
            areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
            idx = int(np.argmax(areas))
            arr = boxes[idx]

        arr = arr.reshape(-1)
        if len(arr) < 4:
            return None

        return tuple(map(float, arr[:4]))

    def get_landmark_and_bbox(
        self,
        img_list: Sequence[str],
        upperbondrange: int = 0,
        avatar_id: Optional[str] = None,
    ):
        """获取每帧人脸关键点和 bbox。"""

        frames = read_imgs(img_list)
        batch_size_fa = 1
        batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]

        coords_list = []
        landmarks = []

        if upperbondrange != 0:
            print(f"get key_landmark and face bounding boxes with the bbox_shift: {upperbondrange}")
        else:
            print("get key_landmark and face bounding boxes with the default value")

        for fb in tqdm(batches):
            frame_np = np.asarray(fb)
            if len(frame_np) == 0:
                continue

            frame = frame_np[0]

            keypoints, scores = self.rtmpose.predict(frame)
            if keypoints is None or len(keypoints) == 0:
                coords_list.append(COORD_PLACEHOLDER)
                landmarks.append(None)
                continue

            face_land_mark = keypoints[0][23:91].astype(np.int32)
            if face_land_mark.shape[0] < 31:
                coords_list.append(COORD_PLACEHOLDER)
                landmarks.append(None)
                continue

            landmarks.append(face_land_mark)

            bbox_list = self.detect_face_bbox(frame_np)

            for f in bbox_list:
                f = self.normalize_detector_bbox(f)
                if f is None:
                    coords_list.append(COORD_PLACEHOLDER)
                    continue

                half_face_coord = face_land_mark[29].copy()

                if upperbondrange != 0:
                    # + 向下，- 向上
                    half_face_coord[1] = int(upperbondrange + half_face_coord[1])

                half_face_dist = int(np.max(face_land_mark[:, 1]) - half_face_coord[1])
                upper_bond = int(half_face_coord[1] - half_face_dist)

                f_landmark = (
                    int(np.min(face_land_mark[:, 0])),
                    int(upper_bond),
                    int(np.max(face_land_mark[:, 0])),
                    int(np.max(face_land_mark[:, 1])),
                )

                x1, y1, x2, y2 = f_landmark

                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                    clipped = safe_clip_bbox(f, frame.shape)
                    if clipped is None:
                        coords_list.append(COORD_PLACEHOLDER)
                    else:
                        coords_list.append(clipped)
                    print("[WARN] bad landmark bbox, fallback detector bbox:", f_landmark, "->", f)
                else:
                    clipped = safe_clip_bbox(f_landmark, frame.shape)
                    coords_list.append(clipped if clipped is not None else COORD_PLACEHOLDER)

        return coords_list, frames

    def get_image_prepare_material(
        self,
        image: np.ndarray,
        face_box,
        upper_boundary_ratio: float = 0.5,
        expand: float = 1.2,
    ):
        """生成 MuseTalk mask 和 crop_box。"""

        clipped_face_box = safe_clip_bbox(face_box, image.shape)
        if clipped_face_box is None:
            return None, None

        face_box = clipped_face_box
        body = Image.fromarray(image[:, :, ::-1])

        x, y, x1, y1 = face_box
        crop_box, _ = get_crop_box(face_box, expand)
        x_s, y_s, x_e, y_e = crop_box

        face_large = body.crop(crop_box)
        ori_shape = face_large.size

        mask_image = self.face_seg(face_large)
        if mask_image is None:
            return None, None

        mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        mask_image = Image.new("L", ori_shape, 0)
        mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

        # 只保留下半部分说话区域
        width, height = mask_image.size
        top_boundary = int(height * upper_boundary_ratio)
        modified_mask_image = Image.new("L", ori_shape, 0)
        modified_mask_image.paste(
            mask_image.crop((0, top_boundary, width, height)),
            (0, top_boundary),
        )

        blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
        mask_array = cv2.GaussianBlur(
            np.array(modified_mask_image),
            (blur_kernel_size, blur_kernel_size),
            0,
        )

        return mask_array, crop_box

    def prepare_input_images(self, file: str, avatar_id: str) -> Tuple[str, List[str]]:
        paths = self.get_avatar_paths(avatar_id)
        save_full_path = paths["full_imgs_path"]

        if os.path.isfile(file):
            if is_video_file(file):
                print(f"input is video: {file}")
                video2imgs(file, save_full_path, ext="png")
            else:
                print(f"input is image: {file}")
                dst_path = os.path.join(save_full_path, os.path.basename(file))
                shutil.copyfile(file, dst_path)

        elif os.path.isdir(file):
            print(f"input is image directory: {file}")
            files = sorted(os.listdir(file))

            for filename in files:
                src_path = os.path.join(file, filename)
                if not os.path.isfile(src_path):
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMG_EXTS:
                    continue

                dst_path = os.path.join(save_full_path, filename)
                shutil.copyfile(src_path, dst_path)

        else:
            raise FileNotFoundError(f"输入文件或目录不存在: {file}")

        input_img_list = list_images(save_full_path)
        if len(input_img_list) == 0:
            raise RuntimeError(f"没有找到有效图片: {save_full_path}")

        return save_full_path, input_img_list

    def build_latents_from_bboxes(self, coord_list, frame_list):
        """根据 bbox 裁剪人脸并生成 VAE latent。为保证和帧数量一致，无效帧用 None 占位。"""

        input_latent_list = []

        for idx, (bbox, frame) in enumerate(tqdm(list(zip(coord_list, frame_list)), desc="build latents")):
            if bbox == COORD_PLACEHOLDER:
                print(f"[WARN] skip latent frame {idx}, no bbox")
                input_latent_list.append(None)
                continue

            crop_frame = safe_crop(frame, bbox)
            if crop_frame is None:
                print(f"[WARN] skip latent frame {idx}, empty crop, bbox={bbox}")
                input_latent_list.append(None)
                continue

            resized_crop_frame = cv2.resize(
                crop_frame,
                (256, 256),
                interpolation=cv2.INTER_LANCZOS4,
            )

            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        return input_latent_list

    def create_musetalk_human(
        self,
        file: str,
        avatar_id: str = "default_avatar",
        bbox_shift: int = 5,
    ):
        """
        创建 MuseTalk 数字人素材。

        输出结构：
        outputs/avatars/<avatar_id>/
            full_imgs/         # 正序 + 倒序循环帧
            mask/              # mask 图
            coords.pkl         # 循环后的 face bbox
            mask_coords.pkl    # 循环后的 mask crop box
            latents.pt         # 循环后的 VAE latent，个别无效帧可能是 None
        """

        paths = self.get_avatar_paths(avatar_id)
        save_full_path, input_img_list = self.prepare_input_images(file, avatar_id)

        print(f"共找到 {len(input_img_list)} 张图片")
        print("extracting landmarks...")

        coord_list, frame_list = self.get_landmark_and_bbox(
            input_img_list,
            upperbondrange=bbox_shift,
            avatar_id=avatar_id,
        )

        print(f"coord num: {len(coord_list)}")
        print(f"frame num: {len(frame_list)}")

        input_latent_list = self.build_latents_from_bboxes(coord_list, frame_list)

        # 做正序 + 倒序循环，和原 MuseTalk avatar 构建方式一致
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        mask_coords_list_cycle = []
        mask_list_cycle = []

        print("building masks...")
        for i, frame in enumerate(tqdm(frame_list_cycle)):
            frame_out_path = os.path.join(save_full_path, f"{i:08d}.png")
            cv2.imwrite(frame_out_path, frame)

            face_box = coord_list_cycle[i]
            if face_box == COORD_PLACEHOLDER:
                mask_coords_list_cycle.append(None)
                mask_list_cycle.append(None)
                print(f"[WARN] skip mask frame {i}, no bbox")
                continue

            mask, crop_box = self.get_image_prepare_material(frame, face_box)
            if mask is None or crop_box is None:
                mask_coords_list_cycle.append(None)
                mask_list_cycle.append(None)
                print(f"[WARN] skip mask frame {i}, face_seg failed or invalid crop")
                continue

            mask_out_file = os.path.join(paths["mask_out_path"], f"{i:08d}.png")
            cv2.imwrite(mask_out_file, mask)
            mask_coords_list_cycle.append(crop_box)
            mask_list_cycle.append(mask)

        with open(paths["mask_coords_path"], "wb") as f:
            pickle.dump(mask_coords_list_cycle, f)

        with open(paths["coords_path"], "wb") as f:
            pickle.dump(coord_list_cycle, f)

        torch.save(input_latent_list_cycle, paths["latents_out_path"])

        print("avatar material build done:", paths["avatar_path"])
        print("valid latent num:", sum(x is not None for x in input_latent_list_cycle))
        print("valid mask num:", sum(x is not None for x in mask_list_cycle))

        return {
            "avatar_id": avatar_id,
            "avatar_path": paths["avatar_path"],
            "full_imgs_path": paths["full_imgs_path"],
            "mask_out_path": paths["mask_out_path"],
            "coords_path": paths["coords_path"],
            "mask_coords_path": paths["mask_coords_path"],
            "latents_out_path": paths["latents_out_path"],
            "input_img_list": input_img_list,
            "coord_list": coord_list,
            "frame_num": len(frame_list),
            "cycle_frame_num": len(frame_list_cycle),
        }

    # 兼容你原来的函数名，避免外部代码调用报错
    def create_musetalk_huma(self, *args, **kwargs):
        return self.create_musetalk_human(*args, **kwargs)


if __name__ == "__main__":
    builder = MusetalkAvatarBuilder(
        config_path="configs/musetalk_v15.yaml",
        output_root="outputs/avatars",
        # 如果 SFDDetector 没有 detect_from_batch/detect，可改为 True
        use_face_alignment=False,
    )

    result = builder.create_musetalk_human(
        file="assets/imgs/3456.png",
        avatar_id="avatar_001",
        bbox_shift=5,
    )

    print(result)
