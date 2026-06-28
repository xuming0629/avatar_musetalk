#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import wave
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

import av
import cv2
import numpy as np
from aiortc import AudioStreamTrack, VideoStreamTrack


class AvatarVideoTrack(VideoStreamTrack):
    """WebRTC 数字人视频轨道。

    - 没有说话帧时：输出 idle_frame / last_frame
    - 数字人回答时：从 queue 里按 fps 输出 MuseTalk 生成帧
    """

    kind = "video"

    def __init__(
        self,
        fps: int = 25,
        width: int = 512,
        height: int = 512,
        max_queue_size: int = 5000,
    ):
        super().__init__()

        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)

        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=max_queue_size
        )

        self.idle_frame: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None

        self._pts = 0
        self._time_base = Fraction(1, self.fps)

    def set_idle_frame(self, frame: np.ndarray):
        frame = self._normalize_frame(frame)
        self.idle_frame = frame
        self.last_frame = frame

    async def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def enqueue_frames(self, frames: List[np.ndarray]):
        """一次性放入本轮回答的全部视频帧。"""

        await self.clear()

        for frame in frames:
            frame = self._normalize_frame(frame)

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            await self.queue.put(frame)

    async def recv(self):
        await asyncio.sleep(1.0 / float(self.fps))

        try:
            frame = self.queue.get_nowait()
            self.last_frame = frame
        except asyncio.QueueEmpty:
            if self.last_frame is not None:
                frame = self.last_frame
            elif self.idle_frame is not None:
                frame = self.idle_frame
            else:
                frame = self._blank_frame()

        video_frame = av.VideoFrame.from_ndarray(
            frame,
            format="bgr24",
        )

        video_frame.pts = self._pts
        video_frame.time_base = self._time_base

        self._pts += 1

        return video_frame

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"invalid video frame shape: {frame.shape}")

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

        return frame

    def _blank_frame(self) -> np.ndarray:
        return np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )


class AvatarAudioTrack(AudioStreamTrack):
    """WebRTC 数字人音频轨道。

    - TTS wav 会被读成 int16 PCM
    - 重采样到 48000 Hz
    - 每 20ms 输出一个 AudioFrame
    """

    kind = "audio"

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        frame_duration_ms: int = 20,
        max_queue_size: int = 10000,
    ):
        super().__init__()

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frame_duration_ms = int(frame_duration_ms)

        self.samples_per_frame = int(
            self.sample_rate * self.frame_duration_ms / 1000
        )

        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=max_queue_size
        )

        self._pts = 0
        self._time_base = Fraction(1, self.sample_rate)

    async def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def enqueue_wav(self, wav_path: str):
        """读取 wav，并切成 WebRTC 音频帧。"""

        await self.clear()

        pcm, sr = self._read_wav_mono_int16(wav_path)

        if sr != self.sample_rate:
            pcm = self._resample_int16(
                pcm,
                src_rate=sr,
                dst_rate=self.sample_rate,
            )

        chunks = self._split_audio_chunks(
            pcm,
            chunk_size=self.samples_per_frame,
        )

        for chunk in chunks:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            await self.queue.put(chunk)

    async def recv(self):
        await asyncio.sleep(self.frame_duration_ms / 1000.0)

        try:
            samples = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            samples = np.zeros(
                (self.samples_per_frame,),
                dtype=np.int16,
            )

        samples = np.asarray(samples, dtype=np.int16)
        samples = np.ascontiguousarray(samples)

        if samples.shape[0] < self.samples_per_frame:
            pad = np.zeros(
                (self.samples_per_frame - samples.shape[0],),
                dtype=np.int16,
            )
            samples = np.concatenate([samples, pad], axis=0)

        elif samples.shape[0] > self.samples_per_frame:
            samples = samples[: self.samples_per_frame]

        frame = av.AudioFrame(
            format="s16",
            layout="mono",
            samples=samples.shape[0],
        )

        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base

        self._pts += samples.shape[0]

        return frame

    @staticmethod
    def _read_wav_mono_int16(wav_path: str):
        wav_path = str(wav_path)

        if not Path(wav_path).exists():
            raise FileNotFoundError(f"wav 不存在: {wav_path}")

        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sample_width != 2:
            raise RuntimeError(
                f"当前只支持 16-bit PCM wav，当前 sample_width={sample_width}"
            )

        audio = np.frombuffer(raw, dtype=np.int16)

        if channels > 1:
            audio = audio.reshape(-1, channels)
            audio = audio.mean(axis=1).astype(np.int16)

        return audio, int(sample_rate)

    @staticmethod
    def _resample_int16(
        audio: np.ndarray,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        if src_rate == dst_rate:
            return audio.astype(np.int16)

        audio = audio.astype(np.float32)

        src_len = audio.shape[0]
        dst_len = int(round(src_len * float(dst_rate) / float(src_rate)))

        if src_len <= 1 or dst_len <= 1:
            return np.zeros((max(dst_len, 1),), dtype=np.int16)

        src_x = np.linspace(0.0, 1.0, num=src_len)
        dst_x = np.linspace(0.0, 1.0, num=dst_len)

        out = np.interp(dst_x, src_x, audio)
        out = np.clip(out, -32768, 32767).astype(np.int16)

        return out

    @staticmethod
    def _split_audio_chunks(
        audio: np.ndarray,
        chunk_size: int,
    ) -> List[np.ndarray]:
        chunks = []

        total = audio.shape[0]

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunks.append(audio[start:end])

        if not chunks:
            chunks.append(
                np.zeros(
                    (chunk_size,),
                    dtype=np.int16,
                )
            )

        return chunks