#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np


class BaseTTSService:
    name = "base"

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


class DummyTTSService(BaseTTSService):
    """生成静音 wav，先跑通 MuseTalk 流程。"""

    name = "dummy"

    def __init__(
        self,
        out_dir: str = "./results/tts",
        seconds: float = 3.0,
        sample_rate: int = 16000,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seconds = float(seconds)
        self.sample_rate = int(sample_rate)

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        out_path = self.out_dir / f"dummy_tts_{int(time.time() * 1000)}.wav"

        samples = int(self.seconds * self.sample_rate)
        audio = np.zeros(samples, dtype=np.int16)

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        return str(out_path)


class EdgeTTSService(BaseTTSService):
    """EdgeTTS 文本转语音。

    需要安装：
        pip install edge-tts

    依赖 ffmpeg 转 wav：
        sudo apt-get install -y ffmpeg
    """

    name = "edgetts"

    def __init__(
        self,
        out_dir: str = "./results/tts",
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: int = 16000,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.sample_rate = int(sample_rate)

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        text = (text or "").strip()

        if not text:
            raise RuntimeError("TTS 文本为空")

        ts = int(time.time() * 1000)

        mp3_path = self.out_dir / f"tts_{ts}.mp3"
        wav_path = self.out_dir / f"tts_{ts}.wav"

        self._run_async(
            self._edge_tts_save(
                text=text,
                mp3_path=str(mp3_path),
            )
        )

        self._convert_to_wav(
            in_path=str(mp3_path),
            out_path=str(wav_path),
        )

        return str(wav_path)

    async def _edge_tts_save(
        self,
        text: str,
        mp3_path: str,
    ):
        try:
            import edge_tts
        except ImportError as e:
            raise ImportError(
                "缺少 edge-tts，请先安装：pip install edge-tts"
            ) from e

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )

        await communicate.save(mp3_path)

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        return asyncio.run(coro)

    def _convert_to_wav(
        self,
        in_path: str,
        out_path: str,
    ):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            out_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )