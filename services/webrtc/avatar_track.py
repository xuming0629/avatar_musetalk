#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import time
from fractions import Fraction
from typing import Optional

import av
import cv2
import numpy as np
from aiortc import VideoStreamTrack


class AvatarVideoTrack(VideoStreamTrack):
    """WebRTC 数字人视频轨道。

    后端不断把 MuseTalk 生成的 RGB frame 放进 queue。
    WebRTC 客户端不断从 recv() 取 frame。
    如果没有新说话帧，就持续输出 idle_frame。
    """

    kind = "video"

    def __init__(
        self,
        fps: int = 25,
        width: int = 512,
        height: int = 512,
    ):
        super().__init__()

        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)

        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=5)

        self.idle_frame: Optional[np.ndarray] = None
        self.last_frame: Optional[np.ndarray] = None

        self._start_time = time.time()
        self._frame_index = 0

    def set_idle_frame(self, frame: np.ndarray):
        """设置待机帧，RGB 格式。"""

        if frame is None:
            return

        frame = self._normalize_frame(frame)
        self.idle_frame = frame
        self.last_frame = frame

    async def push_frame(self, frame: np.ndarray):
        """推入一帧数字人画面，RGB 格式。"""

        if frame is None:
            return

        frame = self._normalize_frame(frame)

        # 如果队列满了，丢掉旧帧，保证实时性
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await self.queue.put(frame)

    async def recv(self):
        """WebRTC 拉流回调。"""

        # 控制 FPS
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
            format="rgb24",
        )

        video_frame.pts = self._frame_index
        video_frame.time_base = Fraction(1, self.fps)

        self._frame_index += 1

        return video_frame

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """统一成 RGB uint8 HWC。"""

        frame = np.asarray(frame)

        if frame.ndim != 3:
            raise ValueError(f"frame shape error: {frame.shape}")

        if frame.shape[2] != 3:
            raise ValueError(f"frame channel error: {frame.shape}")

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 保持比例 resize 到统一尺寸
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