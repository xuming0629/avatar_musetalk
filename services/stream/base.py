#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


class FrameEncoder:
    """把 RGB numpy frame 编码成 JPEG bytes，用于 WebSocket 推流。"""

    @staticmethod
    def rgb_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
        if frame is None:
            raise ValueError("frame is None")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()


class BaseStreamService:
    """推流服务接口。"""

    name = "base"

    def push_frames(self, frames: Iterator[np.ndarray]):
        raise NotImplementedError


class WebRTCStreamService(BaseStreamService):
    """WebRTC 预留实现位置。

    真正上生产建议用 aiortc / LiveKit / Agora 等。
    """

    name = "webrtc"

    def push_frames(self, frames: Iterator[np.ndarray]):
        raise NotImplementedError("请在 services/stream/base.py 中接入 WebRTC 推流逻辑")
