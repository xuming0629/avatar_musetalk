#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MuseTalk 实时数字人服务层。

这个文件只做工程封装，不改你原来的 MuseTalk 底层算法：
- VAE / UNet / PositionalEncoding 继续使用 src.nets.musetalk
- Audio2Feature 继续使用 src.nets.whisper.audio2feature
- get_landmark_and_bbox / get_image 继续使用 src.utils

主要新增：
- AvatarSession：人物缓存，避免每次说话重复做人脸检测和 VAE encode
- RealtimeMuseTalkAvatarService：音频 chunk -> 一帧数字人画面
"""
from __future__ import annotations

import copy
import os
import shutil
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch

from src.utils.blending import get_image
from src.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
from src.utils.utils import get_file_type, get_video_fps
from src.nets.face.parsing.face_parser import FaceParsing
from src.nets.musetalk.unet import PositionalEncoding, UNet
from src.nets.musetalk.vae import VAE
from src.nets.whisper.audio2feature import Audio2Feature

from .types import RealtimeAvatarConfig, StreamFrame


def mkdir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_dir(path: str | Path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_video_frames(video_path: str) -> List[np.ndarray]:
    """读取视频全部帧，返回 RGB。"""
    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()
    return frames


def read_image_as_frame(image_path: str) -> List[np.ndarray]:
    """读取单张图片，返回 RGB list。"""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [img]


def save_frames_to_dir(frames: List[np.ndarray], out_dir: str | Path) -> List[str]:
    """保存 RGB 帧为 PNG，返回路径列表。"""
    clean_dir(out_dir)
    paths = []
    for i, frame in enumerate(frames):
        p = Path(out_dir) / f"{i:08d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
    return paths


def make_cycle_list(items: List):
    if len(items) <= 1:
        return items
    return items + items[::-1]


def to_tensor_audio_feature(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]
    return torch.from_numpy(x)


class AvatarSession:
    """数字人形象缓存。

    build() 只做一次，后续每一帧通过 next() 循环取：
    - 原始帧 ori_frame
    - 人脸 bbox
    - VAE latent
    """

    def __init__(self):
        self.coord_cycle: List[List[int]] = []
        self.frame_cycle: List[np.ndarray] = []
        self.latent_cycle: List[torch.Tensor] = []
        self.index = 0
        self.fps = 25

    def build(
        self,
        avatar_path: str,
        service: "RealtimeMuseTalkAvatarService",
        bbox_shift: int = 0,
        extra_margin: int = 10,
    ):
        file_type = get_file_type(avatar_path)
        if file_type == "video":
            input_fps = get_video_fps(avatar_path)
            if input_fps is not None and input_fps > 0:
                self.fps = int(round(input_fps))
            frames = read_video_frames(avatar_path)
        else:
            frames = read_image_as_frame(avatar_path)

        if not frames:
            raise RuntimeError("没有读取到人物图片/视频帧")

        cache_dir = Path(service.cfg.result_dir) / "avatar_cache" / time.strftime("%Y%m%d_%H%M%S")
        frame_paths = save_frames_to_dir(frames, cache_dir)

        coord_list, frame_list, latent_list = service.prepare_face_latents(
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

    def next(self) -> Tuple[np.ndarray, List[int], torch.Tensor]:
        if not self.frame_cycle:
            raise RuntimeError("AvatarSession 尚未 build")
        idx = self.index % len(self.frame_cycle)
        self.index += 1
        return copy.deepcopy(self.frame_cycle[idx]), self.coord_cycle[idx], self.latent_cycle[idx]



def run_cuda_forward_without_cudnn(fn, device: torch.device):
    """CUDA forward 保护器。

    作用：
        1. CPU 时保持原逻辑；
        2. CUDA 时仍然走 GPU，不会回退 CPU；
        3. 仅在当前 forward 范围内临时关闭 cuDNN；
        4. forward 结束后恢复原来的 cuDNN 设置。

    说明：
        当前环境下部分模型在 cuDNN forward 中可能出现段错误。
        这里用于保护 UNet/VAE 等核心 forward。
    """
    if "cuda" not in str(device):
        with torch.inference_mode():
            return fn()

    old_cudnn_enabled = torch.backends.cudnn.enabled
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic

    try:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.cuda.synchronize()

        with torch.inference_mode():
            result = fn()

        torch.cuda.synchronize()
        return result

    finally:
        torch.backends.cudnn.enabled = old_cudnn_enabled
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic


class RealtimeMuseTalkAvatarService:
    """MuseTalk 实时推理服务。

    设计目标：底层算法不动，只封装成可被 pipeline/TTS/ASR/LLM 调用的 service。
    """

    def __init__(self, cfg: Optional[RealtimeAvatarConfig] = None):
        self.cfg = cfg or RealtimeAvatarConfig()
        mkdir(self.cfg.result_dir)

        self.device = self._resolve_device(self.cfg.device)
        self.weight_dtype = self._resolve_dtype(self.cfg.dtype, self.device)
        self.timesteps = torch.tensor([0], device=self.device, dtype=torch.long)

        print("========== RealtimeMuseTalkAvatarService ==========")
        print("device:", self.device)
        print("weight_dtype:", self.weight_dtype)

        # 全局模型在 service 初始化时加载一次。
        # 注意：必须显式传入 device，避免 service.device 是 cuda，
        # 但模型内部又从其它 config 读成 cpu，导致 CPU/CUDA 混用。
        self.vae = VAE(
            device=str(self.device),
            use_float16=(self.weight_dtype == torch.float16),
        )

        self.unet = UNet(
            device=str(self.device),
            use_float16=(self.weight_dtype == torch.float16),
        )

        # 以 UNet 实际权重 dtype 为准，避免 Half 输入 + Float 权重报错。
        self.weight_dtype = next(self.unet.model.parameters()).dtype

        self.pe = PositionalEncoding(d_model=384)
        self.pe = self.pe.to(device=self.device, dtype=self.weight_dtype)
        self.pe.eval()

        self.fp = FaceParsing(
            left_cheek_width=self.cfg.left_cheek_width,
            right_cheek_width=self.cfg.right_cheek_width,
        )

        self.audio_processor = Audio2Feature(
            device=str(self.device),
        )

        print("unet dtype:", self.weight_dtype)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device):
        if dtype == "fp32":
            return torch.float32
        if dtype == "fp16":
            return torch.float16
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def set_face_parsing_params(self, left_cheek_width: int, right_cheek_width: int):
        self.cfg.left_cheek_width = int(left_cheek_width)
        self.cfg.right_cheek_width = int(right_cheek_width)
        self.fp = FaceParsing(
            left_cheek_width=self.cfg.left_cheek_width,
            right_cheek_width=self.cfg.right_cheek_width,
        )

    def extract_audio_chunks(self, audio_path: str, fps: int):
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        try:
            chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        except TypeError:
            chunks = self.audio_processor.feature2chunks(whisper_feature, fps)
        return chunks

    def prepare_face_latents(self, frame_paths: List[str], bbox_shift: int, extra_margin: int):
        coord_list, frame_list = get_landmark_and_bbox(frame_paths, bbox_shift)
        if coord_list is None or len(coord_list) == 0:
            raise RuntimeError("没有检测到人脸，请检查输入图片/视频")

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

            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

            with torch.no_grad():
                latent = self.vae.get_latents_for_unet(crop_frame)

            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent)

            latent = latent.to(device=self.device, dtype=self.weight_dtype)
            if latent.ndim == 3:
                latent = latent.unsqueeze(0)

            latent_list.append(latent)
            valid_coord_list.append([x1, y1, x2, y2])
            valid_frame_list.append(frame)

        if not latent_list:
            raise RuntimeError("所有帧都没有得到有效人脸区域，请调整 bbox_shift")

        return valid_coord_list, valid_frame_list, latent_list

    @torch.no_grad()
    def infer_one_frame(
        self,
        audio_chunk,
        latent: torch.Tensor,
        ori_frame: np.ndarray,
        bbox: List[int],
        parsing_mode: str = "jaw",
    ) -> np.ndarray:
        """推理单帧数字人画面。

        输入：
            audio_chunk:
                当前视频帧对应的 whisper audio chunk。

            latent:
                当前 avatar 人脸区域对应的 VAE latent。

            ori_frame:
                原始人物帧。

            bbox:
                人脸框 [x1, y1, x2, y2]。

            parsing_mode:
                融合 mask 模式，默认 jaw。

        关键修复：
            1. audio / latent / timestep / PE / UNet 全部统一到 self.device；
            2. dtype 以 UNet 实际权重 dtype 为准；
            3. UNet forward 使用 cuDNN guard，仍然走 CUDA，但临时关闭 cuDNN；
            4. 不直接使用可能混乱的外部 dtype，避免 CPU/CUDA 或 Half/Float 混用。
        """
        # 保证 UNet / PE 在 service 指定设备上。
        self.unet.model.to(self.device)
        self.unet.model.eval()

        self.pe.to(device=self.device, dtype=next(self.unet.model.parameters()).dtype)
        self.pe.eval()

        # 以 UNet 实际参数 dtype 为准。
        weight_dtype = next(self.unet.model.parameters()).dtype
        self.weight_dtype = weight_dtype

        audio_batch = to_tensor_audio_feature(audio_chunk)
        audio_batch = audio_batch.to(
            device=self.device,
            dtype=weight_dtype,
        )

        if audio_batch.ndim == 2:
            audio_batch = audio_batch[None, ...]

        latent_batch = latent.to(
            device=self.device,
            dtype=weight_dtype,
        )

        if latent_batch.ndim == 3:
            latent_batch = latent_batch.unsqueeze(0)

        audio_feature_batch = self.pe(audio_batch)
        audio_feature_batch = audio_feature_batch.to(
            device=self.device,
            dtype=weight_dtype,
        )

        cur_timesteps = torch.full(
            (latent_batch.shape[0],),
            int(self.timesteps[0].item()),
            device=self.device,
            dtype=torch.long,
        )

        def _forward_unet():
            return self.unet.model(
                latent_batch,
                cur_timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample

        pred_latents = run_cuda_forward_without_cudnn(
            _forward_unet,
            self.device,
        )

        recon = self.vae.decode_latents(pred_latents)

        res_frame = recon[0]

        if isinstance(res_frame, torch.Tensor):
            res_frame = res_frame.detach().cpu().numpy()

        res_frame = np.clip(
            res_frame,
            0,
            255,
        ).astype(np.uint8)

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
            fp=self.fp,
        )

        if combine_frame is None:
            combine_frame = ori_frame

        return combine_frame

    def build_avatar_session(self, avatar_path: str, bbox_shift: Optional[int] = None, extra_margin: Optional[int] = None) -> AvatarSession:
        session = AvatarSession()
        session.build(
            avatar_path=avatar_path,
            service=self,
            bbox_shift=self.cfg.bbox_shift if bbox_shift is None else int(bbox_shift),
            extra_margin=self.cfg.extra_margin if extra_margin is None else int(extra_margin),
        )
        return session

    def stream_from_audio_file(
        self,
        avatar_path: str,
        audio_path: str,
        fps: Optional[int] = None,
        bbox_shift: Optional[int] = None,
        extra_margin: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        remove_last_chunk: Optional[bool] = None,
        realtime_sleep: bool = True,
    ) -> Generator[StreamFrame, None, None]:
        """音频文件驱动的实时逐帧生成。

        注意：这里仍然使用完整音频文件提取 chunks，但输出是一帧一帧 yield，适合先跑通实时框架。
        后续真流式 TTS/麦克风时，可以把 audio_chunk queue 接到 infer_one_frame。
        """
        if not audio_path or not Path(audio_path).exists():
            raise RuntimeError(f"音频文件不存在: {audio_path}")
        if not avatar_path or not Path(avatar_path).exists():
            raise RuntimeError(f"人物图片/视频不存在: {avatar_path}")

        avatar = self.build_avatar_session(
            avatar_path=avatar_path,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
        )

        run_fps = int(fps or avatar.fps or self.cfg.fps)
        mode = parsing_mode or self.cfg.parsing_mode
        chunks = self.extract_audio_chunks(audio_path, fps=run_fps)
        if chunks is None or len(chunks) == 0:
            raise RuntimeError("音频特征为空")

        do_remove_last = self.cfg.remove_last_chunk if remove_last_chunk is None else bool(remove_last_chunk)
        if do_remove_last and len(chunks) > 1:
            chunks = chunks[:-1]

        frame_interval = 1.0 / float(run_fps)

        for i, audio_chunk in enumerate(chunks):
            start_time = time.time()
            ori_frame, bbox, latent = avatar.next()
            frame = self.infer_one_frame(
                audio_chunk=audio_chunk,
                latent=latent,
                ori_frame=ori_frame,
                bbox=bbox,
                parsing_mode=mode,
            )
            yield StreamFrame(index=i, frame=frame, pts=i / float(run_fps), metadata={"fps": run_fps})

            if realtime_sleep:
                cost = time.time() - start_time
                sleep_time = frame_interval - cost
                if sleep_time > 0:
                    time.sleep(sleep_time)
    # def stream_from_audio_file_with_session(
    #     self,
    #     avatar_session,
    #     audio_path: str,
    #     fps: Optional[int] = None,
    #     parsing_mode: Optional[str] = None,
    #     remove_last_chunk: Optional[bool] = None,
    #     realtime_sleep: bool = True,
    # ) -> Generator[StreamFrame, None, None]:
    #     """复用已经 build 好的 AvatarSession 进行数字人说话。

    #     用途：
    #         1. 页面先加载数字人，构建 AvatarSession
    #         2. 用户输入问题
    #         3. LLM + TTS 生成回复音频
    #         4. 复用 AvatarSession 直接生成说话帧

    #     好处：
    #         不再重复读取人物图、不再重复做人脸检测、不再重复 VAE encode。
    #     """

    #     if avatar_session is None:
    #         raise RuntimeError("avatar_session 为空，请先加载数字人")

    #     if not audio_path or not Path(audio_path).exists():
    #         raise RuntimeError(f"音频文件不存在: {audio_path}")

    #     run_fps = int(fps or avatar_session.fps or self.cfg.fps)
    #     mode = parsing_mode or self.cfg.parsing_mode

    #     chunks = self.extract_audio_chunks(
    #         audio_path=audio_path,
    #         fps=run_fps,
    #     )

    #     if chunks is None or len(chunks) == 0:
    #         raise RuntimeError("音频特征为空")

    #     do_remove_last = (
    #         self.cfg.remove_last_chunk
    #         if remove_last_chunk is None
    #         else bool(remove_last_chunk)
    #     )

    #     if do_remove_last and len(chunks) > 1:
    #         chunks = chunks[:-1]

    #     frame_interval = 1.0 / float(run_fps)

    #     for i, audio_chunk in enumerate(chunks):
    #         start_time = time.time()

    #         ori_frame, bbox, latent = avatar_session.next()

    #         frame = self.infer_one_frame(
    #             audio_chunk=audio_chunk,
    #             latent=latent,
    #             ori_frame=ori_frame,
    #             bbox=bbox,
    #             parsing_mode=mode,
    #         )

    #         yield StreamFrame(
    #             index=i,
    #             frame=frame,
    #             pts=i / float(run_fps),
    #             metadata={
    #                 "fps": run_fps,
    #                 "mode": mode,
    #                 "audio_path": audio_path,
    #             },
    #         )

    #         if realtime_sleep:
    #             cost = time.time() - start_time
    #             sleep_time = frame_interval - cost

    #             if sleep_time > 0:
    #                 time.sleep(sleep_time)
    
    
    def stream_from_audio_file_with_session(
        self,
        avatar_session,
        audio_path: str,
        fps: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        remove_last_chunk: Optional[bool] = None,
        realtime_sleep: bool = True,
    ):
        """复用已经 build 好的 AvatarSession 进行数字人说话。"""

        if avatar_session is None:
            raise RuntimeError("avatar_session 为空，请先加载数字人")

        if not audio_path or not Path(audio_path).exists():
            raise RuntimeError(f"音频文件不存在: {audio_path}")

        run_fps = int(fps or avatar_session.fps or self.cfg.fps)
        mode = parsing_mode or self.cfg.parsing_mode

        chunks = self.extract_audio_chunks(
            audio_path=audio_path,
            fps=run_fps,
        )

        if chunks is None or len(chunks) == 0:
            raise RuntimeError("音频特征为空")

        do_remove_last = (
            self.cfg.remove_last_chunk
            if remove_last_chunk is None
            else bool(remove_last_chunk)
        )

        if do_remove_last and len(chunks) > 1:
            chunks = chunks[:-1]

        frame_interval = 1.0 / float(run_fps)

        for i, audio_chunk in enumerate(chunks):
            start_time = time.time()

            ori_frame, bbox, latent = avatar_session.next()

            frame = self.infer_one_frame(
                audio_chunk=audio_chunk,
                latent=latent,
                ori_frame=ori_frame,
                bbox=bbox,
                parsing_mode=mode,
            )

            yield StreamFrame(
                index=i,
                frame=frame,
                pts=i / float(run_fps),
                metadata={
                    "fps": run_fps,
                    "mode": mode,
                    "audio_path": audio_path,
                },
            )

            if realtime_sleep:
                cost = time.time() - start_time
                sleep_time = frame_interval - cost

                if sleep_time > 0:
                    time.sleep(sleep_time)