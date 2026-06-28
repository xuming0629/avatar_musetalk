#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WebSocket 推帧服务骨架。

运行：
    PYTHONPATH=. uvicorn tools.realtime_websocket_server:app --host 0.0.0.0 --port 8000

客户端发送 JSON：
{
  "avatar_path": "./assets/3456.png",
  "audio_path": "./assets/data/audio/sun.wav",
  "fps": 25,
  "bbox_shift": 0,
  "extra_margin": 10,
  "parsing_mode": "jaw"
}

服务端逐帧返回 JPEG bytes。
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from services.pipeline import RealtimeDigitalHumanPipeline
from services.stream.base import FrameEncoder
from services.types import RealtimeAvatarConfig

app = FastAPI(title="XumingAvatar Realtime WebSocket")

PIPELINE = RealtimeDigitalHumanPipeline(RealtimeAvatarConfig(device="auto"))


@app.websocket("/ws/avatar")
async def ws_avatar(websocket: WebSocket):
    await websocket.accept()
    try:
        req = await websocket.receive_json()
        avatar_path = req.get("avatar_path")
        audio_path = req.get("audio_path")
        fps = int(req.get("fps", 25))
        bbox_shift = int(req.get("bbox_shift", 0))
        extra_margin = int(req.get("extra_margin", 10))
        parsing_mode = req.get("parsing_mode", "jaw")

        await websocket.send_json({"type": "status", "message": "start"})

        for item in PIPELINE.stream_audio_file(
            avatar_path=avatar_path,
            audio_path=audio_path,
            fps=fps,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            realtime_sleep=True,
        ):
            jpg = FrameEncoder.rgb_to_jpeg_bytes(item.frame)
            await websocket.send_bytes(jpg)

        await websocket.send_json({"type": "status", "message": "done"})

    except WebSocketDisconnect:
        print("[WebSocket] client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
