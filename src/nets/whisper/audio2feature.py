#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : audio2feature.py
# @Time          : 2026-06-24
# @Author        : XuMing
# @description   : Whisper audio feature extractor
"""

import os
import sys
from typing import List, Tuple

import torch
import numpy as np

from src.nets.common.config import (
    load_yaml,
    get_device,
)

from src.nets.whisper.whisper import (
    Whisper,
    ModelDimensions,
)

from src.nets.whisper.transcribe import transcribe

class Audio2Feature:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        whisper_model_type: str = "tiny",
    ):
        self.model_path = model_path
        self.device = device
        self.whisper_model_type = whisper_model_type

        self.model = self.load_model(
            model_path=self.model_path,
            device=self.device,
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/musetalk_v15.yaml",
    ):
        cfg = load_yaml(
            config_path
        )

        wisper_cfg = (
            cfg.get("models", {})
            .get("wisper", {})
        )

        model_path = wisper_cfg.get("path")

        if model_path is None:
            raise ValueError(
                "Config error: models.wisper.path not found"
            )

        whisper_model_type = wisper_cfg.get(
            "type",
            "tiny"
        )

        return cls(
            model_path=model_path,
            device=get_device(config_path),
            whisper_model_type=whisper_model_type,
        )

    def load_model(
        self,
        model_path: str,
        device=None,
    ) -> Whisper:
        if device is None:
            device = self.device

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Whisper model file not found: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        dims = ModelDimensions(
            **checkpoint["dims"]
        )

        model = Whisper(
            dims
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(
            device
        )

        model.eval()

        return model

    def get_sliced_feature(
        self,
        feature_array: np.ndarray,
        vid_idx: int,
        audio_feat_length: List[int] = [2, 2],
        fps: int = 25,
    ) -> Tuple[np.ndarray, List[int]]:

        length = len(
            feature_array
        )

        selected_feature = []
        selected_idx = []

        center_idx = int(
            vid_idx * 50 / fps
        )

        left_idx = (
            center_idx
            - audio_feat_length[0] * 2
        )

        right_idx = (
            center_idx
            + (audio_feat_length[1] + 1) * 2
        )

        for idx in range(
            left_idx,
            right_idx
        ):
            idx = max(
                0,
                idx
            )

            idx = min(
                length - 1,
                idx
            )

            x = feature_array[idx]

            selected_feature.append(
                x
            )

            selected_idx.append(
                idx
            )

        selected_feature = np.concatenate(
            selected_feature,
            axis=0
        )

        selected_feature = selected_feature.reshape(
            -1,
            384
        )

        return selected_feature, selected_idx

    def get_sliced_feature_sparse(
        self,
        feature_array: np.ndarray,
        vid_idx: int,
        audio_feat_length: List[int] = [2, 2],
        fps: int = 25,
    ) -> Tuple[np.ndarray, List[int]]:

        length = len(
            feature_array
        )

        selected_feature = []
        selected_idx = []

        for dt in range(
            -audio_feat_length[0],
            audio_feat_length[1] + 1
        ):
            left_idx = int(
                (vid_idx + dt) * 50 / fps
            )

            if left_idx < 1 or left_idx > length - 1:
                left_idx = max(
                    0,
                    left_idx
                )

                left_idx = min(
                    length - 1,
                    left_idx
                )

                x = feature_array[left_idx]
                x = x[np.newaxis, :, :]

                x = np.repeat(
                    x,
                    2,
                    axis=0
                )

                selected_feature.append(
                    x
                )

                selected_idx.append(
                    left_idx
                )

                selected_idx.append(
                    left_idx
                )

            else:
                x = feature_array[
                    left_idx - 1:left_idx + 1
                ]

                selected_feature.append(
                    x
                )

                selected_idx.append(
                    left_idx - 1
                )

                selected_idx.append(
                    left_idx
                )

        selected_feature = np.concatenate(
            selected_feature,
            axis=0
        )

        selected_feature = selected_feature.reshape(
            -1,
            384
        )

        return selected_feature, selected_idx

    def feature2chunks(
        self,
        feature_array: np.ndarray,
        fps: int,
        audio_feat_length: List[int] = [2, 2],
    ) -> List[np.ndarray]:

        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps

        i = 0

        print(
            f"video in {fps} FPS, audio idx in 50FPS"
        )

        while True:
            start_idx = int(
                i * whisper_idx_multiplier
            )

            selected_feature, _ = self.get_sliced_feature(
                feature_array=feature_array,
                vid_idx=i,
                audio_feat_length=audio_feat_length,
                fps=fps,
            )

            whisper_chunks.append(
                selected_feature
            )

            i += 1

            if start_idx > len(feature_array):
                break

        return whisper_chunks

    def audio2feat(
        self,
        audio_path: str,
    ) -> np.ndarray:

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        result = transcribe(self.model,
            audio_path
        )

        embed_list = []

        for emb in result["segments"]:
            encoder_embeddings = emb[
                "encoder_embeddings"
            ]

            encoder_embeddings = encoder_embeddings.transpose(
                0,
                2,
                1,
                3
            )

            encoder_embeddings = encoder_embeddings.squeeze(
                0
            )

            start_idx = int(
                emb["start"]
            )

            end_idx = int(
                emb["end"]
            )

            emb_end_idx = int(
                (end_idx - start_idx) / 2
            )

            embed_list.append(
                encoder_embeddings[:emb_end_idx]
            )

        if len(embed_list) == 0:
            raise RuntimeError(
                f"No whisper feature extracted from: {audio_path}"
            )

        concatenated_array = np.concatenate(
            embed_list,
            axis=0
        )

        return concatenated_array

