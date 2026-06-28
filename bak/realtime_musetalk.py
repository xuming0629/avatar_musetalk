#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# ============================================================
# 如果你要强制 CPU，就打开这一行
# 如果你要实时，建议注释掉这一行，然后用 CUDA
# ============================================================
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import sys
import copy
import time
import shutil
import imageio
import subprocess
from pathlib import Path
from argparse import Namespace
from typing import List, Tuple, Optional

import gradio as gr
import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# 项目工具函数
# ============================================================

from src.utils.blending import get_image
from src.utils.utils import get_file_type, get_video_fps
from src.utils.preprocessing import (
    get_landmark_and_bbox,
    coord_placeholder,
)


# ============================================================
# 模型
# ============================================================

from src.nets.face.parsing.face_parser import FaceParsing
from src.nets.face.dwpose.dwpose import RTMPose
from src.nets.face.s3fd.sfd_detector import SFDDetector
from src.nets.musetalk.vae import VAE
from src.nets.musetalk.unet import UNet, PositionalEncoding
from src.nets.face.alignment.face_alignment import FaceAlignment
from src.nets.whisper.audio2feature import Audio2Feature


# ============================================================
# Runtime 配置
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

TIMESTEPS = torch.tensor([0], device=DEVICE, dtype=torch.long)

print("========== Runtime ==========")
print("device:", DEVICE)
print("weight_dtype:", WEIGHT_DTYPE)


# ============================================================
# 全局模型，只加载一次
# ============================================================

print("========== Loading global models ==========")

vae = VAE()
unet = UNet()
pe = PositionalEncoding(d_model=384)

fa = FaceAlignment()
fp = FaceParsing()

# 如果你后续要换成 dwpose / s3fd，可以打开
# dwpose = RTMPose()
# sf_detect = SFDDetector()

audio_processor = Audio2Feature()

print("========== Models loaded ==========")


# ============================================================
# 参数构造
# ============================================================

def build_args(
    bbox_shift: int = 0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
):
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
# 基础工具
# ============================================================

def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_video_frames(video_path: str) -> List[np.ndarray]:
    """
    读取视频所有帧，返回 RGB 格式。
    """
    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()
    return frames


def read_image_as_frame(image_path: str) -> List[np.ndarray]:
    """
    读取单张图片，返回 RGB 格式列表。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [img]


def save_frames_to_dir(frames: List[np.ndarray], out_dir: str):
    """
    保存 RGB 帧为 PNG。
    """
    clean_dir(out_dir)
    paths = []

    for i, frame in enumerate(frames):
        p = os.path.join(out_dir, f"{i:08d}.png")
        cv2.imwrite(p, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(p)

    return paths


def make_cycle_list(items: List):
    """
    MuseTalk 常见做法：
    视频帧不足时，做正序 + 反序循环。
    """
    if len(items) <= 1:
        return items
    return items + items[::-1]


def to_tensor_audio_feature(x):
    """
    把 audio chunk 转成 torch tensor。
    目标 shape 通常是 [B, T, 384]。
    """
    if isinstance(x, torch.Tensor):
        return x

    x = np.asarray(x)

    if x.ndim == 2:
        x = x[None, ...]

    x = torch.from_numpy(x)
    return x


def batch_iter(audio_chunks, latent_cycle, batch_size: int):
    """
    离线 batch 生成器。
    """
    total = len(audio_chunks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        audio_batch = []
        latent_batch = []

        for i in range(start, end):
            audio_batch.append(audio_chunks[i])
            latent_batch.append(latent_cycle[i % len(latent_cycle)])

        audio_batch = np.stack(audio_batch, axis=0)

        if isinstance(latent_batch[0], torch.Tensor):
            latent_batch = torch.cat(latent_batch, dim=0)
        else:
            latent_batch = torch.from_numpy(np.concatenate(latent_batch, axis=0))

        yield start, audio_batch, latent_batch


def run_cmd(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def mux_audio_video(
    silent_video_path: str,
    audio_path: str,
    output_path: str,
):
    """
    把无声视频和音频合成最终视频。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", silent_video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    run_cmd(cmd)


def frames_to_video(
    frame_dir: str,
    fps: int,
    output_path: str,
):
    """
    PNG 序列帧转无声视频。
    """
    pattern = os.path.join(frame_dir, "%08d.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-r", str(fps),
        "-i", pattern,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    run_cmd(cmd)


def fix_last_green_frame(frame_dir: str):
    """
    如果最后一帧明显偏绿，用倒数第二帧替换。
    防止尾部 audio chunk 或融合异常导致最后一帧绿屏。
    """
    paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(paths) < 2:
        return

    last_path = paths[-1]
    prev_path = paths[-2]

    img = cv2.imread(last_path)
    if img is None:
        shutil.copy(prev_path, last_path)
        return

    b, g, r = cv2.split(img)
    green_score = (
        g.astype(np.float32) - np.maximum(r, b).astype(np.float32)
    ).mean()

    if green_score > 30:
        print(f"[WARN] last frame looks green, replace with previous frame: {last_path}")
        shutil.copy(prev_path, last_path)


# ============================================================
# 音频特征
# ============================================================

def extract_audio_chunks(audio_path: str, fps: int):
    """
    提取 Whisper 音频特征，并切成视频帧对应的 chunk。
    """
    whisper_feature = audio_processor.audio2feat(audio_path)

    try:
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature,
            fps=fps,
        )
    except TypeError:
        whisper_chunks = audio_processor.feature2chunks(
            whisper_feature,
            fps,
        )

    return whisper_chunks


# ============================================================
# 人脸坐标和 latent 准备
# ============================================================

def prepare_face_latents(
    frame_paths: List[str],
    bbox_shift: int,
    extra_margin: int,
):
    """
    根据帧路径检测人脸框，并构造 VAE latent。

    返回：
    - coord_list
    - frame_list
    - latent_list
    """

    coord_list, frame_list = get_landmark_and_bbox(frame_paths, bbox_shift)

    if coord_list is None or len(coord_list) == 0:
        raise RuntimeError("没有检测到人脸，请检查输入图片/视频。")

    latent_list = []
    valid_coord_list = []
    valid_frame_list = []

    for idx, bbox in enumerate(coord_list):
        if bbox == coord_placeholder:
            print(f"[WARN] frame {idx}: no face detected, skip")
            continue

        frame = frame_list[idx]
        if frame is None:
            print(f"[WARN] frame {idx}: frame is None, skip")
            continue

        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2 + int(extra_margin))

        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] frame {idx}: invalid bbox {(x1, y1, x2, y2)}, skip")
            continue

        crop_frame = frame[y1:y2, x1:x2]

        if crop_frame is None or crop_frame.size == 0:
            print(f"[WARN] frame {idx}: empty crop, skip")
            continue

        crop_frame = cv2.resize(
            crop_frame,
            (256, 256),
            interpolation=cv2.INTER_LANCZOS4,
        )

        with torch.no_grad():
            latent = vae.get_latents_for_unet(crop_frame)

        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent)

        latent = latent.to(device=DEVICE, dtype=WEIGHT_DTYPE)

        # 确保 latent 是 [1, C, H, W]
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)

        latent_list.append(latent)
        valid_coord_list.append([x1, y1, x2, y2])
        valid_frame_list.append(frame)

    if len(latent_list) == 0:
        raise RuntimeError("所有帧都没有得到有效人脸区域，请调整 bbox_shift。")

    return valid_coord_list, valid_frame_list, latent_list


# ============================================================
# 实时 Avatar 缓存
# ============================================================

class AvatarSession:
    """
    实时数字人 Avatar 缓存。

    作用：
    - 人脸检测只做一次
    - VAE latent 只做一次
    - 实时推理时循环取 frame / bbox / latent
    """

    def __init__(self):
        self.coord_cycle = []
        self.frame_cycle = []
        self.latent_cycle = []
        self.index = 0
        self.fps = 25

    def build(
        self,
        video_path: str,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        fps: int = 25,
    ):
        self.fps = fps

        file_type = get_file_type(video_path)

        if file_type == "video":
            input_fps = get_video_fps(video_path)
            if input_fps is not None and input_fps > 0:
                self.fps = int(round(input_fps))

            frames = read_video_frames(video_path)
        else:
            frames = read_image_as_frame(video_path)

        if len(frames) == 0:
            raise RuntimeError("没有读取到人物图片/视频帧。")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tmp_dir = os.path.join("./results/output", f"avatar_cache_{timestamp}")
        mkdir(tmp_dir)

        frame_paths = save_frames_to_dir(frames, tmp_dir)

        coord_list, frame_list, latent_list = prepare_face_latents(
            frame_paths=frame_paths,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
        )

        self.coord_cycle = make_cycle_list(coord_list)
        self.frame_cycle = make_cycle_list(frame_list)
        self.latent_cycle = make_cycle_list(latent_list)
        self.index = 0

        print("[AvatarSession] frames:", len(self.frame_cycle))
        print("[AvatarSession] latents:", len(self.latent_cycle))
        print("[AvatarSession] fps:", self.fps)

    def next(self):
        if len(self.frame_cycle) == 0:
            raise RuntimeError("AvatarSession 还没有 build。")

        idx = self.index % len(self.frame_cycle)
        self.index += 1

        ori_frame = copy.deepcopy(self.frame_cycle[idx])
        bbox = self.coord_cycle[idx]
        latent = self.latent_cycle[idx]

        return ori_frame, bbox, latent


# ============================================================
# 实时单帧推理
# ============================================================

@torch.no_grad()
def infer_one_realtime_frame(
    audio_chunk,
    latent,
    ori_frame,
    bbox,
    parsing_mode="jaw",
):
    """
    输入一个 audio chunk，输出一帧数字人画面。
    """

    audio_batch = to_tensor_audio_feature(audio_chunk)
    audio_batch = audio_batch.to(device=DEVICE, dtype=WEIGHT_DTYPE)

    if audio_batch.ndim == 2:
        audio_batch = audio_batch[None, ...]

    latent_batch = latent.to(device=DEVICE, dtype=WEIGHT_DTYPE)

    if latent_batch.ndim == 3:
        latent_batch = latent_batch.unsqueeze(0)

    audio_feature_batch = pe(audio_batch)

    cur_timesteps = torch.full(
        (latent_batch.shape[0],),
        int(TIMESTEPS[0].item()),
        device=DEVICE,
        dtype=torch.long,
    )

    pred_latents = unet.model(
        latent_batch,
        cur_timesteps,
        encoder_hidden_states=audio_feature_batch,
    ).sample

    recon = vae.decode_latents(pred_latents)

    res_frame = recon[0]

    if isinstance(res_frame, torch.Tensor):
        res_frame = res_frame.detach().cpu().numpy()

    res_frame = np.clip(res_frame, 0, 255).astype(np.uint8)

    x1, y1, x2, y2 = map(int, bbox)

    res_frame = cv2.resize(
        res_frame,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )

    combine_frame = get_image(
        ori_frame,
        res_frame,
        [x1, y1, x2, y2],
        mode=parsing_mode,
        fp=fp,
    )

    if combine_frame is None:
        combine_frame = ori_frame

    return combine_frame


# ============================================================
# 实时流式推理：音频文件驱动
# ============================================================

def realtime_musetalk_stream(
    audio_path: str,
    video_path: str,
    bbox_shift: int = 0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    fps: int = 25,
    remove_last_chunk: bool = True,
):
    """
    最小实时版：
    - 输入还是音频文件
    - 输出变成逐帧 yield
    - 不保存 PNG
    - 不合成 mp4
    """

    if audio_path is None or not os.path.exists(audio_path):
        raise gr.Error("音频文件不存在")

    if video_path is None or not os.path.exists(video_path):
        raise gr.Error("视频/图片文件不存在")

    print("[Realtime] build avatar session...")

    avatar = AvatarSession()
    avatar.build(
        video_path=video_path,
        bbox_shift=bbox_shift,
        extra_margin=extra_margin,
        fps=fps,
    )

    real_fps = avatar.fps if avatar.fps > 0 else fps

    print("[Realtime] extract audio chunks...")

    audio_chunks = extract_audio_chunks(audio_path, fps=real_fps)

    if audio_chunks is None or len(audio_chunks) == 0:
        raise RuntimeError("音频特征为空。")

    if remove_last_chunk and len(audio_chunks) > 1:
        audio_chunks = audio_chunks[:-1]

    print("[Realtime] audio chunks:", len(audio_chunks))
    print("[Realtime] fps:", real_fps)
    print("[Realtime] start streaming...")

    frame_interval = 1.0 / float(real_fps)

    for i, audio_chunk in enumerate(audio_chunks):
        start_time = time.time()

        ori_frame, bbox, latent = avatar.next()

        frame = infer_one_realtime_frame(
            audio_chunk=audio_chunk,
            latent=latent,
            ori_frame=ori_frame,
            bbox=bbox,
            parsing_mode=parsing_mode,
        )

        # Gradio Image 输出一般用 RGB
        yield frame

        cost = time.time() - start_time
        sleep_time = frame_interval - cost

        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================
# 原离线 MuseTalk 推理：保留
# ============================================================

def run_musetalk_inference(
    audio_path: str,
    video_path: str,
    args: Namespace,
):
    """
    离线版本 MuseTalk 推理流程。
    输出 mp4。
    """

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    result_dir = args.result_dir
    mkdir(result_dir)

    work_dir = os.path.join(result_dir, f"work_{timestamp}")
    input_frame_dir = os.path.join(work_dir, "input_frames")
    result_frame_dir = os.path.join(work_dir, "result_frames")

    mkdir(work_dir)

    silent_video_path = os.path.join(work_dir, "silent.mp4")

    if args.output_vid_name:
        output_video_path = os.path.join(result_dir, args.output_vid_name)
    else:
        output_video_path = os.path.join(result_dir, f"result_{timestamp}.mp4")

    # --------------------------------------------------------
    # 1. 读取输入帧
    # --------------------------------------------------------
    print("[STEP] 读取输入视频/图片...")

    file_type = get_file_type(video_path)

    if file_type == "video":
        input_fps = get_video_fps(video_path)
        if input_fps is not None and input_fps > 0:
            args.fps = int(round(input_fps))

        frames = read_video_frames(video_path)
    else:
        frames = read_image_as_frame(video_path)

    if len(frames) == 0:
        raise RuntimeError("没有读取到任何输入帧。")

    print(f"[INFO] input frames: {len(frames)}")
    print(f"[INFO] fps: {args.fps}")

    frame_paths = save_frames_to_dir(frames, input_frame_dir)

    # --------------------------------------------------------
    # 2. 音频特征
    # --------------------------------------------------------
    print("[STEP] 提取音频特征...")

    audio_chunks = extract_audio_chunks(audio_path, fps=args.fps)

    if audio_chunks is None or len(audio_chunks) == 0:
        raise RuntimeError("音频特征为空，请检查 Audio2Feature。")

    # 防止最后一个尾部 chunk 异常
    if len(audio_chunks) > 1:
        audio_chunks = audio_chunks[:-1]

    print(f"[INFO] audio chunks: {len(audio_chunks)}")

    # --------------------------------------------------------
    # 3. 人脸检测 + latent
    # --------------------------------------------------------
    print("[STEP] 检测人脸并编码 latent...")

    coord_list, frame_list, latent_list = prepare_face_latents(
        frame_paths=frame_paths,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
    )

    coord_cycle = make_cycle_list(coord_list)
    frame_cycle = make_cycle_list(frame_list)
    latent_cycle = make_cycle_list(latent_list)

    # --------------------------------------------------------
    # 4. 开始 UNet 推理
    # --------------------------------------------------------
    print("[STEP] 执行 MuseTalk 推理...")

    clean_dir(result_frame_dir)

    total_chunks = len(audio_chunks)

    with torch.no_grad():
        pbar = tqdm(
            batch_iter(audio_chunks, latent_cycle, args.batch_size),
            total=(total_chunks + args.batch_size - 1) // args.batch_size,
        )

        for batch_start, audio_batch, latent_batch in pbar:
            audio_batch = to_tensor_audio_feature(audio_batch)
            audio_batch = audio_batch.to(device=DEVICE, dtype=WEIGHT_DTYPE)

            latent_batch = latent_batch.to(device=DEVICE, dtype=WEIGHT_DTYPE)

            audio_feature_batch = pe(audio_batch)

            cur_timesteps = torch.full(
                (latent_batch.shape[0],),
                int(TIMESTEPS[0].item()),
                device=DEVICE,
                dtype=torch.long,
            )

            pred_latents = unet.model(
                latent_batch,
                cur_timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample

            recon = vae.decode_latents(pred_latents)

            for j in range(recon.shape[0]):
                global_idx = batch_start + j

                if global_idx >= total_chunks:
                    break

                frame_idx = global_idx % len(frame_cycle)

                bbox = coord_cycle[frame_idx]
                ori_frame = copy.deepcopy(frame_cycle[frame_idx])

                x1, y1, x2, y2 = map(int, bbox)

                res_frame = recon[j]

                if isinstance(res_frame, torch.Tensor):
                    res_frame = res_frame.detach().cpu().numpy()

                res_frame = np.clip(res_frame, 0, 255).astype(np.uint8)

                res_frame = cv2.resize(
                    res_frame,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LINEAR,
                )

                combine_frame = get_image(
                    ori_frame,
                    res_frame,
                    [x1, y1, x2, y2],
                    mode=args.parsing_mode,
                    fp=fp,
                )

                if combine_frame is None:
                    combine_frame = ori_frame

                save_path = os.path.join(result_frame_dir, f"{global_idx:08d}.png")

                # combine_frame 是 RGB，cv2.imwrite 需要 BGR
                cv2.imwrite(
                    save_path,
                    cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR),
                )

    # --------------------------------------------------------
    # 5. 帧合成视频
    # --------------------------------------------------------
    print("[STEP] 合成无声视频...")

    fix_last_green_frame(result_frame_dir)

    frames_to_video(
        frame_dir=result_frame_dir,
        fps=args.fps,
        output_path=silent_video_path,
    )

    # --------------------------------------------------------
    # 6. 合成音频
    # --------------------------------------------------------
    print("[STEP] 合成最终带音频视频...")

    mux_audio_video(
        silent_video_path=silent_video_path,
        audio_path=audio_path,
        output_path=output_video_path,
    )

    print("[INFO] output:", output_video_path)

    return output_video_path


# ============================================================
# Gradio 离线 inference 入口
# ============================================================

def inference(
    audio_path,
    video_path,
    bbox_shift,
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
):
    """
    Gradio 离线调用入口。
    返回 mp4 路径。
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

    global fp
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    print("========== Inference args ==========")
    print(args)
    print("audio_path:", audio_path)
    print("video_path:", video_path)

    try:
        output_path = run_musetalk_inference(
            audio_path=audio_path,
            video_path=video_path,
            args=args,
        )
        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"推理失败: {str(e)}")


# ============================================================
# Gradio 实时 inference 入口
# ============================================================

def realtime_inference(
    audio_path,
    video_path,
    bbox_shift,
    extra_margin,
    parsing_mode,
    fps,
    left_cheek_width,
    right_cheek_width,
):
    """
    Gradio 实时调用入口。
    逐帧 yield。
    """

    global fp
    fp = FaceParsing(
        left_cheek_width=int(left_cheek_width),
        right_cheek_width=int(right_cheek_width),
    )

    for frame in realtime_musetalk_stream(
        audio_path=audio_path,
        video_path=video_path,
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        fps=int(fps),
    ):
        yield frame


# ============================================================
# Gradio 实时 Demo
# ============================================================

def main_realtime_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# 实时数字人 Demo - MuseTalk Streaming")

        gr.Markdown(
            """
            这个版本是最小实时版：
            - 输入还是音频文件
            - 但是画面会逐帧输出
            - 不等待全部视频合成完成
            - 后续可以接 TTS / ASR / LLM / WebRTC
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="输入音频",
                    type="filepath",
                )

                video_input = gr.File(
                    label="输入人物图片/视频",
                    type="filepath",
                )

                bbox_shift = gr.Slider(
                    minimum=-20,
                    maximum=20,
                    value=0,
                    step=1,
                    label="bbox_shift",
                )

                extra_margin = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=10,
                    step=1,
                    label="extra_margin",
                )

                parsing_mode = gr.Dropdown(
                    choices=["jaw", "raw", "face"],
                    value="jaw",
                    label="parsing_mode",
                )

                fps = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=25,
                    step=1,
                    label="FPS",
                )

                left_cheek_width = gr.Slider(
                    minimum=0,
                    maximum=150,
                    value=90,
                    step=1,
                    label="left_cheek_width",
                )

                right_cheek_width = gr.Slider(
                    minimum=0,
                    maximum=150,
                    value=90,
                    step=1,
                    label="right_cheek_width",
                )

                start_btn = gr.Button("开始实时生成")

            with gr.Column():
                output_image = gr.Image(
                    label="实时数字人画面",
                    type="numpy",
                )

        start_btn.click(
            fn=realtime_inference,
            inputs=[
                audio_input,
                video_input,
                bbox_shift,
                extra_margin,
                parsing_mode,
                fps,
                left_cheek_width,
                right_cheek_width,
            ],
            outputs=output_image,
        )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


# ============================================================
# Gradio 离线 Demo
# ============================================================

def main_offline_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# MuseTalk 离线视频生成 Demo")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="输入音频",
                    type="filepath",
                )

                video_input = gr.File(
                    label="输入人物图片/视频",
                    type="filepath",
                )

                bbox_shift = gr.Slider(
                    minimum=-20,
                    maximum=20,
                    value=0,
                    step=1,
                    label="bbox_shift",
                )

                extra_margin = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=10,
                    step=1,
                    label="extra_margin",
                )

                parsing_mode = gr.Dropdown(
                    choices=["jaw", "raw", "face"],
                    value="jaw",
                    label="parsing_mode",
                )

                left_cheek_width = gr.Slider(
                    minimum=0,
                    maximum=150,
                    value=90,
                    step=1,
                    label="left_cheek_width",
                )

                right_cheek_width = gr.Slider(
                    minimum=0,
                    maximum=150,
                    value=90,
                    step=1,
                    label="right_cheek_width",
                )

                start_btn = gr.Button("开始生成视频")

            with gr.Column():
                output_video = gr.Video(
                    label="输出视频",
                )

        start_btn.click(
            fn=inference,
            inputs=[
                audio_input,
                video_input,
                bbox_shift,
                extra_margin,
                parsing_mode,
                left_cheek_width,
                right_cheek_width,
            ],
            outputs=output_video,
        )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


# ============================================================
# 命令行离线推理
# ============================================================

def main_cli():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        type=str,
        default="./assets/data/audio/sun.wav",
        help="input audio path",
    )

    parser.add_argument(
        "--video",
        type=str,
        default="./assets/3456.png",
        help="input video/image path",
    )

    parser.add_argument("--result_dir", default="./results/output", type=str)
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--bbox_shift", default=0, type=int)
    parser.add_argument("--extra_margin", default=10, type=int)
    parser.add_argument("--parsing_mode", default="jaw", choices=["jaw", "raw", "face"])
    parser.add_argument("--left_cheek_width", default=90, type=int)
    parser.add_argument("--right_cheek_width", default=90, type=int)
    parser.add_argument("--output_vid_name", default="", type=str)

    args = parser.parse_args()

    infer_args = build_args(
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    infer_args.result_dir = args.result_dir
    infer_args.fps = args.fps
    infer_args.batch_size = args.batch_size
    infer_args.output_vid_name = args.output_vid_name

    global fp
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    output_path = run_musetalk_inference(
        audio_path=args.audio,
        video_path=args.video,
        args=infer_args,
    )

    print("output:", output_path)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    """
    三种运行方式：

    1. 实时 Gradio：
       python realtime_musetalk.py realtime

    2. 离线 Gradio：
       python realtime_musetalk.py offline_gradio

    3. 命令行生成 mp4：
       python realtime_musetalk.py cli --audio xxx.wav --video xxx.png
    """

    mode = "realtime"

    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    if mode == "realtime":
        main_realtime_gradio()

    elif mode == "offline_gradio":
        main_offline_gradio()

    elif mode == "cli":
        main_cli()

    else:
        print("Unknown mode:", mode)
        print("Usage:")
        print("  python realtime_musetalk.py realtime")
        print("  python realtime_musetalk.py offline_gradio")
        print("  python realtime_musetalk.py cli --audio xxx.wav --video xxx.png")