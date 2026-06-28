#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


class BaseMicrophoneService:
    """麦克风服务接口。

    当前先不强绑定 pyaudio/sounddevice，避免环境依赖。
    Gradio 里上传/录制的 audio filepath 可以直接给 ASR 使用。
    后续如果要真流式麦克风，再实现 stream_pcm。
    """

    name = "base"

    def stream_pcm(self) -> Iterator[np.ndarray]:
        raise NotImplementedError


class DummyMicrophoneService(BaseMicrophoneService):
    """占位麦克风：不输出实时 PCM。"""

    name = "dummy"

    def stream_pcm(self) -> Iterator[np.ndarray]:
        if False:
            yield np.zeros(160, dtype=np.float32)
        return


class SoundDeviceMicrophoneService(BaseMicrophoneService):
    """sounddevice 麦克风预留实现位置。"""

    name = "sounddevice"

    def __init__(self, sample_rate: int = 16000, block_size: int = 320):
        self.sample_rate = sample_rate
        self.block_size = block_size

    def stream_pcm(self) -> Iterator[np.ndarray]:
        raise NotImplementedError("请在 services/mic/base.py 中接入 sounddevice 实时录音逻辑")
