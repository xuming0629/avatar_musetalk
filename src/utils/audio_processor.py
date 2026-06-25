#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : audio_processor.py
# @Time          : 2026-06-24
# @Author        : XuMing
# @description   : MuseTalk Whisper AudioProcessor with default config support
"""

import math
import os
from typing import Optional, Tuple, List

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor

from src.nets.common.config import (
    load_yaml,
    resolve_path,
    get_device,
)


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"


def get_whisper_feature_extractor_path(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
) -> str:
    """
    从配置文件读取 Whisper feature extractor 路径。

    兼容两种配置：

    models:
      whisper:
        path: models/whisper

    或：

    models:
      wisper:
        path: models/whisper
    """

    cfg = load_yaml(config_path)

    models_cfg = cfg.get("models", {})

    whisper_cfg = models_cfg.get("whisper", None)

    if whisper_cfg is None:
        whisper_cfg = models_cfg.get("wisper", None)

    if whisper_cfg is None:
        raise ValueError(
            "Config error: models.whisper or models.wisper not found"
        )

    feature_extractor_path = whisper_cfg.get("feature_extractor_path", None)

    if feature_extractor_path is None:
        feature_extractor_path = whisper_cfg.get("path", None)

    if feature_extractor_path is None:
        raise ValueError(
            "Config error: models.whisper.path or models.wisper.path not found"
        )

    feature_extractor_path = resolve_path(
        feature_extractor_path,
        project_root,
    )

    return feature_extractor_path


class AudioProcessor:
    """
    MuseTalk Whisper AudioProcessor。

    支持默认直接声明：

        audio_processor = AudioProcessor()

    等价于：

        audio_processor = AudioProcessor.from_config("configs/musetalk_v15.yaml")

    也支持直接传路径：

        audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
    """

    def __init__(
        self,
        feature_extractor_path: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config_path = config_path
        self.project_root = project_root

        if feature_extractor_path is None:
            feature_extractor_path = get_whisper_feature_extractor_path(
                config_path=config_path,
                project_root=project_root,
            )
        else:
            feature_extractor_path = resolve_path(
                feature_extractor_path,
                project_root,
            )

        if device is None:
            device = get_device(
                config_path=config_path,
            )

        self.feature_extractor_path = feature_extractor_path
        self.device = str(device)

        print(
            f"[AudioProcessor] load feature extractor from: {self.feature_extractor_path}"
        )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.feature_extractor_path,
        )

        print(
            f"[AudioProcessor] load success, device={self.device}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        保留 from_config 写法，兼容旧代码。
        """

        return cls(
            feature_extractor_path=None,
            config_path=config_path,
            project_root=project_root,
            device=device,
        )

    def get_audio_feature(
        self,
        wav_path: str,
        start_index: int = 0,
        weight_dtype=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        加载音频并提取 Whisper input_features。

        返回:
            features:
                List[torch.Tensor]
                每个元素 shape 一般是 [1, 80, 3000]

            librosa_length:
                原始音频采样点数量
        """

        if not os.path.exists(wav_path):
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}"
            )

        librosa_output, sampling_rate = librosa.load(
            wav_path,
            sr=16000,
        )

        if sampling_rate != 16000:
            raise RuntimeError(
                f"Sampling rate error, expected 16000, got {sampling_rate}"
            )

        if start_index > 0:
            librosa_output = librosa_output[start_index:]

        # Split audio into 30s segments
        segment_length = 30 * sampling_rate

        segments = [
            librosa_output[i:i + segment_length]
            for i in range(
                0,
                len(librosa_output),
                segment_length,
            )
        ]

        features = []

        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).input_features

            if weight_dtype is not None:
                audio_feature = audio_feature.to(
                    dtype=weight_dtype,
                )

            features.append(
                audio_feature,
            )

        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ):
        """
        将 Whisper input_features 转为 MuseTalk 每帧 audio prompt。

        官方 MuseTalk 推理里一般这样调用：

            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)

            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )

        返回:
            audio_prompts:
                torch.Tensor, shape [T, C, 384]
                常见是 [num_frames, 50, 384]
        """

        if whisper is None:
            raise ValueError(
                "whisper is None. Please pass WhisperModel / Whisper encoder model."
            )

        if device is None:
            device = self.device

        if weight_dtype is None:
            weight_dtype = torch.float32

        audio_feature_length_per_frame = 2 * (
            audio_padding_length_left
            + audio_padding_length_right
            + 1
        )

        whisper_feature = []

        # Process multiple 30s mel input features
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(
                device=device,
                dtype=weight_dtype,
            )

            with torch.no_grad():
                audio_feats = whisper.encoder(
                    input_feature,
                    output_hidden_states=True,
                ).hidden_states

            audio_feats = torch.stack(
                audio_feats,
                dim=2,
            )

            whisper_feature.append(
                audio_feats,
            )

        whisper_feature = torch.cat(
            whisper_feature,
            dim=1,
        )

        # Trim the last segment to remove padding
        sr = 16000
        audio_fps = 50
        fps = int(fps)

        whisper_idx_multiplier = audio_fps / fps

        num_frames = math.floor(
            (librosa_length / sr) * fps,
        )

        actual_length = math.floor(
            (librosa_length / sr) * audio_fps,
        )

        whisper_feature = whisper_feature[
            :,
            :actual_length,
            ...,
        ]

        # Calculate padding amount
        padding_nums = math.ceil(
            whisper_idx_multiplier,
        )

        # Add padding at start and end
        whisper_feature = torch.cat(
            [
                torch.zeros_like(
                    whisper_feature[
                        :,
                        :padding_nums * audio_padding_length_left,
                    ]
                ),
                whisper_feature,
                torch.zeros_like(
                    whisper_feature[
                        :,
                        :padding_nums * 3 * audio_padding_length_right,
                    ]
                ),
            ],
            dim=1,
        )

        audio_prompts = []

        for frame_index in range(num_frames):
            try:
                audio_index = math.floor(
                    frame_index * whisper_idx_multiplier,
                )

                audio_clip = whisper_feature[
                    :,
                    audio_index: audio_index + audio_feature_length_per_frame,
                ]

                assert audio_clip.shape[1] == audio_feature_length_per_frame

                audio_prompts.append(
                    audio_clip,
                )

            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(
                    f"num frames: {num_frames}, "
                    f"fps: {fps}, "
                    f"whisper_idx_multiplier: {whisper_idx_multiplier}"
                )
                print(
                    f"frame_index: {frame_index}, "
                    f"audio_index: {audio_index}-"
                    f"{audio_index + audio_feature_length_per_frame}"
                )

                raise

        audio_prompts = torch.cat(
            audio_prompts,
            dim=0,
        )

        audio_prompts = rearrange(
            audio_prompts,
            "b c h w -> b (c h) w",
        )

        return audio_prompts

    def __call__(
        self,
        wav_path: str,
        weight_dtype=None,
    ):
        return self.get_audio_feature(
            wav_path,
            weight_dtype=weight_dtype,
        )


if __name__ == "__main__":
    audio_processor = AudioProcessor()

    wav_path = "./2.wav"

    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(
        wav_path,
    )

    print("audio feature segments:", len(audio_feature))

    if len(audio_feature) > 0:
        print("first audio feature shape:", audio_feature[0].shape)

    print("librosa_feature_length:", librosa_feature_length)