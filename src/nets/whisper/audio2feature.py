#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : audio2feature.py
# @Time          : 2026-06-24
# @Author        : XuMing
# @description   : Whisper audio feature extractor with default config support
"""

import os
from typing import List, Tuple, Optional

import torch
import numpy as np

from src.nets.common.config import (
    load_yaml,
    resolve_path,
    get_device,
)

from src.nets.whisper.whisper import (
    Whisper,
    ModelDimensions,
)

from src.nets.whisper.transcribe import transcribe


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"


def get_whisper_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
):
    """
    从配置文件读取 Whisper / Wisper 参数。

    兼容两种 key：

    models:
      wisper:
        path: models/whisper/tiny.pt
        type: tiny

    或：

    models:
      whisper:
        path: models/whisper/tiny.pt
        type: tiny
    """

    cfg = load_yaml(config_path)

    models_cfg = cfg.get("models", {})

    # 兼容你当前写法 wisper
    whisper_cfg = models_cfg.get("wisper", None)

    # 也兼容标准拼写 whisper
    if whisper_cfg is None:
        whisper_cfg = models_cfg.get("whisper", None)

    if whisper_cfg is None:
        raise ValueError(
            "Config error: models.wisper or models.whisper not found"
        )

    model_path = whisper_cfg.get("path", None)

    if model_path is None:
        raise ValueError(
            "Config error: models.wisper.path or models.whisper.path not found"
        )

    model_path = resolve_path(
        model_path,
        project_root,
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Whisper model file not found: {model_path}"
        )

    whisper_model_type = whisper_cfg.get(
        "type",
        "tiny",
    )

    return {
        "model_path": model_path,
        "whisper_model_type": whisper_model_type,
    }


class Audio2Feature:
    """
    Whisper audio feature extractor.

    支持默认直接声明：

        audio_processor = Audio2Feature()

    等价于：

        audio_processor = Audio2Feature.from_config("configs/musetalk_v15.yaml")

    也支持直接传模型路径：

        audio_processor = Audio2Feature(model_path="models/whisper/tiny.pt")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
        whisper_model_type: Optional[str] = None,
        eval_mode: bool = True,
    ):
        self.config_path = config_path
        self.project_root = project_root

        if model_path is None:
            whisper_cfg = get_whisper_config(
                config_path=config_path,
                project_root=project_root,
            )

            model_path = whisper_cfg["model_path"]

            if whisper_model_type is None:
                whisper_model_type = whisper_cfg["whisper_model_type"]

        else:
            model_path = resolve_path(
                model_path,
                project_root,
            )

            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Whisper model file not found: {model_path}"
                )

            if whisper_model_type is None:
                whisper_model_type = "tiny"

        if device is None:
            device = get_device(
                config_path=config_path,
            )

        self.model_path = model_path
        self.device = str(device)
        self.whisper_model_type = whisper_model_type

        print(f"[Audio2Feature] load whisper from: {self.model_path}")

        self.model = self.load_model(
            model_path=self.model_path,
            device=self.device,
        )

        if eval_mode:
            self.model.eval()

        print(
            f"[Audio2Feature] load success: {self.model_path}, "
            f"device={self.device}, "
            f"type={self.whisper_model_type}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
        whisper_model_type: Optional[str] = None,
    ):
        """
        保留 from_config 写法，兼容旧代码。
        """

        return cls(
            model_path=None,
            config_path=config_path,
            project_root=project_root,
            device=device,
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
            map_location="cpu",
        )

        dims = ModelDimensions(
            **checkpoint["dims"],
        )

        model = Whisper(
            dims,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
        )

        model = model.to(
            device,
        )

        model.eval()
        model.requires_grad_(False)

        return model

    def get_sliced_feature(
        self,
        feature_array: np.ndarray,
        vid_idx: int,
        audio_feat_length: Optional[List[int]] = None,
        fps: int = 25,
    ) -> Tuple[np.ndarray, List[int]]:

        if audio_feat_length is None:
            audio_feat_length = [2, 2]

        length = len(feature_array)

        selected_feature = []
        selected_idx = []

        center_idx = int(
            vid_idx * 50 / fps,
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
            right_idx,
        ):
            idx = max(
                0,
                idx,
            )

            idx = min(
                length - 1,
                idx,
            )

            x = feature_array[idx]

            selected_feature.append(
                x,
            )

            selected_idx.append(
                idx,
            )

        selected_feature = np.concatenate(
            selected_feature,
            axis=0,
        )

        selected_feature = selected_feature.reshape(
            -1,
            384,
        )

        return selected_feature, selected_idx

    def get_sliced_feature_sparse(
        self,
        feature_array: np.ndarray,
        vid_idx: int,
        audio_feat_length: Optional[List[int]] = None,
        fps: int = 25,
    ) -> Tuple[np.ndarray, List[int]]:

        if audio_feat_length is None:
            audio_feat_length = [2, 2]

        length = len(feature_array)

        selected_feature = []
        selected_idx = []

        for dt in range(
            -audio_feat_length[0],
            audio_feat_length[1] + 1,
        ):
            left_idx = int(
                (vid_idx + dt) * 50 / fps,
            )

            if left_idx < 1 or left_idx > length - 1:
                left_idx = max(
                    0,
                    left_idx,
                )

                left_idx = min(
                    length - 1,
                    left_idx,
                )

                x = feature_array[left_idx]
                x = x[np.newaxis, :, :]

                x = np.repeat(
                    x,
                    2,
                    axis=0,
                )

                selected_feature.append(
                    x,
                )

                selected_idx.append(
                    left_idx,
                )

                selected_idx.append(
                    left_idx,
                )

            else:
                x = feature_array[
                    left_idx - 1:left_idx + 1
                ]

                selected_feature.append(
                    x,
                )

                selected_idx.append(
                    left_idx - 1,
                )

                selected_idx.append(
                    left_idx,
                )

        selected_feature = np.concatenate(
            selected_feature,
            axis=0,
        )

        selected_feature = selected_feature.reshape(
            -1,
            384,
        )

        return selected_feature, selected_idx

    def feature2chunks(
        self,
        feature_array: np.ndarray,
        fps: int = 25,
        audio_feat_length: Optional[List[int]] = None,
        audio_padding_length_left: Optional[int] = None,
        audio_padding_length_right: Optional[int] = None,
        use_sparse: bool = False,
    ) -> List[np.ndarray]:
        """
        将 whisper feature 切成每一帧对应的 audio chunk。

        参数:
            feature_array:
                audio2feat 输出。

            fps:
                视频 FPS。

            audio_feat_length:
                [left, right]，默认 [2, 2]。

            audio_padding_length_left / right:
                兼容官方 MuseTalk Gradio 调用。
                如果传了这两个参数，会覆盖 audio_feat_length。

            use_sparse:
                是否使用 sparse 采样版本。
        """

        if audio_padding_length_left is not None or audio_padding_length_right is not None:
            left = 2 if audio_padding_length_left is None else int(audio_padding_length_left)
            right = 2 if audio_padding_length_right is None else int(audio_padding_length_right)
            audio_feat_length = [left, right]

        if audio_feat_length is None:
            audio_feat_length = [2, 2]

        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps

        i = 0

        print(
            f"video in {fps} FPS, audio idx in 50FPS"
        )

        while True:
            start_idx = int(
                i * whisper_idx_multiplier,
            )

            if use_sparse:
                selected_feature, _ = self.get_sliced_feature_sparse(
                    feature_array=feature_array,
                    vid_idx=i,
                    audio_feat_length=audio_feat_length,
                    fps=fps,
                )
            else:
                selected_feature, _ = self.get_sliced_feature(
                    feature_array=feature_array,
                    vid_idx=i,
                    audio_feat_length=audio_feat_length,
                    fps=fps,
                )

            whisper_chunks.append(
                selected_feature,
            )

            i += 1

            if start_idx > len(feature_array):
                break

        return whisper_chunks

    def audio2feat(
        self,
        audio_path: str,
    ) -> np.ndarray:
        """
        从音频提取 whisper encoder embedding。

        输出:
            np.ndarray, shape: [T, ?, 384] 或拼接后的 whisper 特征。
        """

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        result = transcribe(
            self.model,
            audio_path,
        )

        embed_list = []

        for emb in result["segments"]:
            encoder_embeddings = emb[
                "encoder_embeddings"
            ]

            # [?, ?, ?, ?] -> MuseTalk 所需格式
            encoder_embeddings = encoder_embeddings.transpose(
                0,
                2,
                1,
                3,
            )

            encoder_embeddings = encoder_embeddings.squeeze(
                0,
            )

            start_idx = int(
                emb["start"],
            )

            end_idx = int(
                emb["end"],
            )

            emb_end_idx = int(
                (end_idx - start_idx) / 2,
            )

            embed_list.append(
                encoder_embeddings[:emb_end_idx],
            )

        if len(embed_list) == 0:
            raise RuntimeError(
                f"No whisper feature extracted from: {audio_path}"
            )

        concatenated_array = np.concatenate(
            embed_list,
            axis=0,
        )

        return concatenated_array

    def get_audio_feature(
        self,
        audio_path: str,
    ):
        """
        兼容官方 MuseTalk Gradio 写法：

            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)

        你的 transcribe 版本里没有单独返回 librosa_length，
        这里用 len(feature_array) 作为近似长度返回。
        """

        feature_array = self.audio2feat(
            audio_path,
        )

        librosa_length = len(
            feature_array,
        )

        return feature_array, librosa_length

    def get_whisper_chunk(
        self,
        whisper_input_features: np.ndarray,
        device=None,
        weight_dtype=None,
        whisper=None,
        librosa_length=None,
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ):
        """
        兼容官方 MuseTalk Gradio 写法：

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

        这里 whisper/device/weight_dtype/librosa_length 为了兼容接口保留，
        当前实现不额外使用。
        """

        whisper_chunks = self.feature2chunks(
            feature_array=whisper_input_features,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )

        if weight_dtype is None:
            weight_dtype = torch.float32

        tensor_chunks = []

        for chunk in whisper_chunks:
            chunk_tensor = torch.from_numpy(
                chunk,
            ).float()

            if device is not None:
                chunk_tensor = chunk_tensor.to(
                    device=device,
                    dtype=weight_dtype,
                )

            tensor_chunks.append(
                chunk_tensor,
            )

        return tensor_chunks

    def __call__(
        self,
        audio_path: str,
    ) -> np.ndarray:
        return self.audio2feat(
            audio_path,
        )


if __name__ == "__main__":
    audio_processor = Audio2Feature()

    audio_path = "assets/test.mp3"

    feature = audio_processor.audio2feat(
        audio_path,
    )

    print(
        "feature:",
        feature.shape,
        feature.dtype,
    )

    chunks = audio_processor.feature2chunks(
        feature,
        fps=25,
    )

    print(
        "chunks:",
        len(chunks),
        chunks[0].shape if len(chunks) > 0 else None,
    )