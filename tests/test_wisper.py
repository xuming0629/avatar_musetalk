#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_wisper.py
# @Time          : 2026-06-22 16:04:13
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   :
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""


import os
import sys
import torch
import numpy as np
import soundfile as sf
import time
from typing import Optional, Union
from model import   Whisper, ModelDimensions  # 确保安装 openai-whisper

sys.path.append("..")

# --------------------- 设备检测 ---------------------
def get_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            print("[INFO] Using device: npu:0")
            return torch.device("npu:0")
    except ImportError:
        pass

    if torch.cuda.is_available():
        print("[INFO] Using device: cuda")
        return torch.device("cuda")
    else:
        print("[INFO] Using device: cpu")
        return torch.device("cpu")


# --------------------- 模型加载 ---------------------
def load_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> Whisper:
    """
    仅从本地加载 Whisper 模型，不下载
    """
    if device is None:
        device = get_device()

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[INFO] Whisper model loaded from: {model_path}")
    return model


# --------------------- 音频特征类 ---------------------
class Audio2Feature:
    def __init__(self,
                 whisper_model_type="tiny",
                 model_path="tiny.pt"):
        self.whisper_model_type = whisper_model_type
        self.device = get_device()
        self.model = load_model(model_path, self.device)

    # -------------------- 提取音频特征 --------------------
    def audio2feat(self, audio_path):
        """
        提取 Whisper 模型的 encoder embedding 特征
        """
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result['segments']:
            encoder_embeddings = emb['encoder_embeddings']
            encoder_embeddings = encoder_embeddings.transpose(0, 2, 1, 3).squeeze(0)
            start_idx = int(emb['start'])
            end_idx = int(emb['end'])
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        concatenated_array = np.concatenate(embed_list, axis=0)
        return concatenated_array

    # -------------------- 按索引取特征块 --------------------
    def get_sliced_feature(self, feature_array, vid_idx, audio_feat_length=[2, 2], fps=25):
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, min(idx, length - 1))
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)
        return selected_feature, selected_idx

    def get_sliced_feature_sparse(self, feature_array, vid_idx, audio_feat_length=[2, 2], fps=25):
        length = len(feature_array)
        selected_feature, selected_idx = [], []

        for dt in range(-audio_feat_length[0], audio_feat_length[1] + 1):
            left_idx = int((vid_idx + dt) * 50 / fps)
            if left_idx < 1 or left_idx > length - 1:
                left_idx = max(0, min(left_idx, length - 1))
                x = feature_array[left_idx]
                x = np.repeat(x[np.newaxis, :, :], 2, axis=0)
                selected_feature.append(x)
                selected_idx += [left_idx, left_idx]
            else:
                x = feature_array[left_idx - 1:left_idx + 1]
                selected_feature.append(x)
                selected_idx += [left_idx - 1, left_idx]

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps, batch_size, audio_feat_length=[2, 2], start=0):
        whisper_chunks = []
        i = 0
        for _ in range(batch_size):
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=feature_array,
                vid_idx=i + start,
                audio_feat_length=audio_feat_length,
                fps=fps
            )
            whisper_chunks.append(selected_feature)
            i += 1
        return whisper_chunks


# --------------------- 测试 ---------------------
if __name__ == "__main__":
    audio_processor = Audio2Feature(model_path="tiny.pt")
    audio_path = "./test.mp3"
    array = audio_processor.audio2feat(audio_path)
    print("Feature shape:", array.shape)

    fps = 25
    whisper_idx_multiplier = 50. / fps
    i = 0
    while True:
        start_idx = int(i * whisper_idx_multiplier)
        selected_feature, selected_idx = audio_processor.get_sliced_feature(array, vid_idx=i, audio_feat_length=[2, 2], fps=fps)
        print(f"video idx {i}, audio idx {selected_idx}, shape {selected_feature.shape}")
        i += 1
        if start_idx > len(array):
            break
