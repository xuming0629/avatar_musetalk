#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : audio.py
# @Time          : 2026-06-24 09:18:10
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : Whisper 音频预处理模块
# @Company       : 2026 XuMing. All Rights Reserved.

功能：
    1. 读取音频并转为 16k 单声道 waveform
    2. 对音频进行补零或裁剪
    3. 加载 Mel Filter Bank
    4. 计算 Whisper 所需的 Log-Mel 特征

输出：
    log_mel_spectrogram: shape = (N_MELS, N_FRAMES)
"""

import os
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

from src.nets.whisper import (
    N_SAMPLES,
    N_FFT,
    N_MELS,
    HOP_LENGTH,
    SAMPLE_RATE,
    MEL_FILTERS_PATH
)


def load_audio(
    file: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    读取音频文件，并转换为 Whisper 所需的 waveform。

    处理流程：
        1. 使用 ffmpeg 解码音频
        2. 转换为单声道
        3. 重采样到指定采样率
        4. 转换为 float32，范围归一化到 [-1, 1]

    Args:
        file: 音频文件路径
        sr: 目标采样率，默认 SAMPLE_RATE

    Returns:
        np.ndarray:
            单声道音频波形，shape = (num_samples,)
    """
    try:
        out, _ = (
            ffmpeg.input(
                file,
                threads=0,
            )
            .output(
                "-",
                format="s16le",
                acodec="pcm_s16le",
                ac=1,
                ar=sr,
            )
            .run(
                cmd=[
                    "ffmpeg",
                    "-nostdin",
                ],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Failed to load audio: {e.stderr.decode()}"
        ) from e

    audio = (
        np.frombuffer(
            out,
            np.int16,
        )
        .flatten()
        .astype(np.float32)
        / 32768.0
    )

    return audio


def pad_or_trim(
    array,
    length: int = N_SAMPLES,
    *,
    axis: int = -1,
):
    """
    将输入音频裁剪或补零到固定长度。

    Whisper 默认以 30 秒为一个 chunk：
        N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH

    如果长度大于 length：
        直接裁剪

    如果长度小于 length：
        末尾补零

    Args:
        array:
            输入音频数组，支持 np.ndarray 或 torch.Tensor
        length:
            目标长度，默认 N_SAMPLES
        axis:
            需要裁剪或补零的维度

    Returns:
        与输入类型一致的固定长度数组
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis,
                index=torch.arange(
                    length,
                    device=array.device,
                ),
            )

        if array.shape[axis] < length:
            pad_widths = [
                (0, 0)
            ] * array.ndim

            pad_widths[axis] = (
                0,
                length - array.shape[axis],
            )

            array = F.pad(
                array,
                [
                    pad
                    for sizes in pad_widths[::-1]
                    for pad in sizes
                ],
            )

    else:
        if array.shape[axis] > length:
            array = array.take(
                indices=range(length),
                axis=axis,
            )

        if array.shape[axis] < length:
            pad_widths = [
                (0, 0)
            ] * array.ndim

            pad_widths[axis] = (
                0,
                length - array.shape[axis],
            )

            array = np.pad(
                array,
                pad_widths,
            )

    return array


@lru_cache(maxsize=None)
def mel_filters(
    device,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """
    加载 Mel Filter Bank。

    Mel Filter Bank 用于将 STFT 得到的频谱能量
    映射到 Mel 频率空间。

    当前项目只支持 Whisper 默认的 80 维 Mel 特征。

    Args:
        device:
            目标设备，例如 cpu 或 cuda
        n_mels:
            Mel 通道数，默认 N_MELS

    Returns:
        torch.Tensor:
            Mel 滤波矩阵
    """
    assert n_mels == 80, (
        f"Unsupported n_mels: {n_mels}"
    )


    with np.load(MEL_FILTERS_PATH) as f:
        filters = torch.from_numpy(
            f[f"mel_{n_mels}"]
        ).to(device)

    return filters


def log_mel_spectrogram(
    audio: Union[
        str,
        np.ndarray,
        torch.Tensor,
    ],
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """
    计算 Whisper Log-Mel 频谱特征。

    输入可以是：
        1. 音频文件路径
        2. np.ndarray waveform
        3. torch.Tensor waveform

    处理流程：
        waveform
            -> STFT
            -> Power Spectrum
            -> Mel Filter
            -> Log-Mel
            -> Whisper 归一化

    Args:
        audio:
            音频路径或 waveform
        n_mels:
            Mel 通道数，默认 N_MELS

    Returns:
        torch.Tensor:
            Log-Mel 特征，shape = (N_MELS, frames)

    默认 Whisper 输入：
        shape = (80, 3000)
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(
                audio
            )

        audio = torch.from_numpy(
            audio
        )

    window = torch.hann_window(
        N_FFT
    ).to(
        audio.device
    )

    stft = torch.stft(
        audio,
        N_FFT,
        HOP_LENGTH,
        window=window,
        return_complex=True,
    )

    magnitudes = (
        stft[:, :-1].abs() ** 2
    )

    filters = mel_filters(
        audio.device,
        n_mels,
    )

    mel_spec = (
        filters @ magnitudes
    )

    log_spec = torch.clamp(
        mel_spec,
        min=1e-10,
    ).log10()

    log_spec = torch.maximum(
        log_spec,
        log_spec.max() - 8.0,
    )

    log_spec = (
        log_spec + 4.0
    ) / 4.0

    return log_spec