#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : __init__.py
# @Time          : 2026-06-24 09:14:21
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TODO
# @Company       : 2026 XuMing. All Rights Reserved.
"""
from src.nets.common.config import load_yaml


def exact_div(x: int, y: int) -> int:
    if x % y != 0:
        raise ValueError(
            f"{x} is not divisible by {y}"
        )
    return x // y


def _load_audio_cfg(
    config_path: str = "configs/musetalk_v15.yaml",
):
    cfg = load_yaml(config_path)

    wisper_cfg = (
        cfg.get("models", {})
           .get("wisper", {})
    )

    audio_cfg = wisper_cfg.get("audio")

    if audio_cfg is None:
        audio_cfg = {}

    return audio_cfg


_AUDIO_CFG = _load_audio_cfg()


MEL_FILTERS_PATH = _AUDIO_CFG.get(
    "mel_filters",
    "models/whisper/mel_filters.npz"
)


SAMPLE_RATE = _AUDIO_CFG.get(
    "sample_rate",
    16000,
)

N_FFT = _AUDIO_CFG.get(
    "n_fft",
    400,
)

HOP_LENGTH = _AUDIO_CFG.get(
    "hop_length",
    160,
)

N_MELS = _AUDIO_CFG.get(
    "n_mels",
    80,
)

CHUNK_LENGTH = _AUDIO_CFG.get(
    "chunk_length",
    30,
)

N_SAMPLES = (
    CHUNK_LENGTH
    * SAMPLE_RATE
)

N_FRAMES = exact_div(
    N_SAMPLES,
    HOP_LENGTH,
)