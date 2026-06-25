#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

####################################
# 底层基础模型类
from src.nets.face.dwpose.dwpose import RTMPose
from src.nets.face.parsing.face_parser import FaceParsing
from src.nets.face.s3fd.sfd_detector import SFDDetector
from src.nets.musetalk.vae import VAE
from src.nets.musetalk.unet import UNet
from src.nets.face.alignment.face_alignment import FaceAlignment
from src.nets.common.config import get_device
####################################


def read_imgs(img_list):
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
    vid_path,
    save_path,
    ext="png",
    cut_frame=10000000,
):
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


def is_video_file(file_path):
    video_exts = [".mp4", ".mkv", ".flv", ".avi", ".mov"]
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_exts


def safe_crop(frame, bbox):
    """
    安全裁剪 bbox，自动处理：
    - float bbox 转 int
    - bbox 越界
    - 空 crop
    """

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


class MusetalkAvatarBuilder:
    def __init__(
        self,
        config_path: str,
        output_root: str = "outputs/avatars",
        use_face_alignment: bool = True,
    ):
        self.config_path = config_path
        self.output_root = output_root

        self.device = get_device(config_path)

        self.rtmpose = RTMPose.from_config(config_path)

        # 推荐使用 FaceAlignment，因为你原来的逻辑调用的是 detect_from_batch
        if use_face_alignment:
            self.face_detector = FaceAlignment.from_config(config_path)
        else:
            self.face_detector = SFDDetector.from_config(config_path)

        self.vae = VAE.from_config(config_path)

        # self.unet = UNet.from_config(config_path)
        # self.face_parser = FaceParsing.from_config(config_path)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        print("current_dir:", current_dir)
        print("parent_dir:", parent_dir)

    def detect_face_bbox(self, frame_np):
        """
        统一封装人脸检测。

        优先兼容：
        1. FaceAlignment.detect_from_batch
        2. SFDDetector.detect_from_batch
        3. SFDDetector.detect
        """

        if hasattr(self.face_detector, "detect_from_batch"):
            return self.face_detector.detect_from_batch(frame_np)

        if hasattr(self.face_detector, "detect"):
            bboxes = []

            for img in frame_np:
                bbox = self.face_detector.detect(img)
                bboxes.append(bbox)

            return bboxes

        raise AttributeError(
            "当前 face_detector 没有 detect_from_batch 或 detect 方法，请检查 SFDDetector / FaceAlignment 接口。"
        )

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

            if len(frame_np) == 0:
                continue

            frame = frame_np[0]

            # RTMPose 关键点检测
            keypoints, scores = self.rtmpose.predict(frame)

            if keypoints is None or len(keypoints) == 0:
                coords_list.append(coord_placeholder)
                landmarks.append(None)
                continue

            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)

            landmarks.append(face_land_mark)

            # 人脸检测 bbox
            bbox_list = self.detect_face_bbox(frame_np)

            for j, f in enumerate(bbox_list):
                if f is None:
                    coords_list.append(coord_placeholder)
                    continue

                # 有些检测器返回 [[x1,y1,x2,y2,score]]，这里兼容一下
                f = np.asarray(f)

                if f.ndim == 2:
                    if len(f) == 0:
                        coords_list.append(coord_placeholder)
                        continue

                    # 默认取第一个人脸，也可以按面积最大选择
                    f = f[0]

                if len(f) >= 4:
                    f = f[:4]
                else:
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
                    coords_list.append(tuple(map(float, f)))

                    w = f[2] - f[0]
                    h = f[3] - f[1]

                    print("error landmark bbox:", f_landmark)
                    print("fallback detector bbox:", f)
                    print("fallback bbox size:", w, h)
                else:
                    coords_list.append(f_landmark)

        return coords_list, frames

    def prepare_input_images(
        self,
        file: str,
        avatar_id: str,
    ):
        """
        将视频 / 单图 / 图片目录统一整理到 full_imgs 目录。
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
            glob.glob(os.path.join(save_full_path, "*.png"))
            + glob.glob(os.path.join(save_full_path, "*.jpg"))
            + glob.glob(os.path.join(save_full_path, "*.jpeg"))
            + glob.glob(os.path.join(save_full_path, "*.bmp"))
            + glob.glob(os.path.join(save_full_path, "*.webp"))
        )

        if len(input_img_list) == 0:
            raise RuntimeError(f"没有找到有效图片: {save_full_path}")

        return save_full_path, input_img_list

    def build_latents_from_bboxes(
        self,
        coord_list,
        frame_list,
    ):
        """
        根据 bbox 裁剪人脸区域，并送入 VAE 得到 latent。
        """

        input_latent_list = []

        coord_placeholder = (0.0, 0.0, 0.0, 0.0)

        for idx, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list))):
            if bbox == coord_placeholder:
                print(f"[WARN] skip frame {idx}, no valid bbox")
                continue

            crop_frame = safe_crop(frame, bbox)

            if crop_frame is None:
                print(f"[WARN] skip frame {idx}, empty crop, bbox={bbox}")
                continue

            resized_crop_frame = cv2.resize(
                crop_frame,
                (256, 256),
                interpolation=cv2.INTER_LANCZOS4,
            )

            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        return input_latent_list

    def create_musetalk_huma(
        self,
        file: str,
        avatar_id: str = "default_avatar",
        bbox_shift: int = 5,
    ):
        """
        创建 MuseTalk 数字人素材。

        支持：
        1. 输入视频文件：自动抽帧
        2. 输入单张图片：复制到目标目录
        3. 输入图片文件夹：复制 png/jpg/jpeg/bmp/webp 图片
        """

        save_full_path, input_img_list = self.prepare_input_images(
            file=file,
            avatar_id=avatar_id,
        )

        print(f"共找到 {len(input_img_list)} 张图片")
        print("extracting landmarks...")

        coord_list, frame_list = self.get_landmark_and_bbox(
            input_img_list,
            upperbondrange=bbox_shift,
            avatar_id=avatar_id,
        )

        print(f"coord num: {len(coord_list)}")
        print(f"frame num: {len(frame_list)}")

        input_latent_list = self.build_latents_from_bboxes(
            coord_list=coord_list,
            frame_list=frame_list,
        )

        print(f"latent num: {len(input_latent_list)}")

        return {
            "avatar_id": avatar_id,
            "save_full_path": save_full_path,
            "input_img_list": input_img_list,
            "coord_list": coord_list,
            "frame_list": frame_list,
            "input_latent_list": input_latent_list,
        }
        
        
        builder = MusetalkAvatarBuilder(
    config_path="configs/config.yaml",
    output_root="outputs/avatars",
    use_face_alignment=True,
)

result = builder.create_musetalk_huma(
    file="data/test.mp4",
    avatar_id="avatar_001",
    bbox_shift=5,
)

print(result["avatar_id"])
print(result["save_full_path"])
print(len(result["input_img_list"]))
print(len(result["coord_list"]))
print(len(result["input_latent_list"]))