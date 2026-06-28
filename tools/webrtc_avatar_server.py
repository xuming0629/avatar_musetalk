#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebRTC 音视频实时数字人服务。

功能：
1. 浏览器建立 WebRTC，接收 video + audio 两条 track
2. 上传 avatar，后端加载并缓存 AvatarSession
3. 输入文本问题
4. 后端 LLM -> TTS
5. MuseTalk 生成视频帧
6. TTS wav 解码成 PCM
7. 视频帧和音频帧同时通过 WebRTC 推给浏览器

运行：
    export MOONSHOT_API_KEY="你的 key"

    PYTHONPATH=. python tools/webrtc_av_avatar_server.py \
      --host 0.0.0.0 \
      --port 8000 \
      --device cpu

GPU：
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python tools/webrtc_av_avatar_server.py \
      --host 0.0.0.0 \
      --port 8000 \
      --device cuda
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional

import aiofiles
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from services.pipeline_v2 import RealtimeDigitalHumanPipeline
from services.types import RealtimeAvatarConfig
from services.webrtc import AvatarAudioTrack, AvatarVideoTrack


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="XumingAvatar WebRTC AV Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 全局状态
# ============================================================

PCS = set()

PIPELINE: Optional[RealtimeDigitalHumanPipeline] = None
VIDEO_TRACK: Optional[AvatarVideoTrack] = None
AUDIO_TRACK: Optional[AvatarAudioTrack] = None

CURRENT_AVATAR_PATH: Optional[str] = None

UPLOAD_DIR = Path("./results/webrtc/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INFER_LOCK: Optional[asyncio.Lock] = None


# ============================================================
# Request Model
# ============================================================

class OfferRequest(BaseModel):
    sdp: str
    type: str


class AskRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


# ============================================================
# 初始化
# ============================================================

def init_runtime(
    device: str = "cpu",
    fps: int = 25,
    bbox_shift: int = 0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    width: int = 512,
    height: int = 512,
):
    global PIPELINE, VIDEO_TRACK, AUDIO_TRACK

    if PIPELINE is not None and VIDEO_TRACK is not None and AUDIO_TRACK is not None:
        return

    cfg = RealtimeAvatarConfig(
        device=device,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        left_cheek_width=90,
        right_cheek_width=90,
        config_path="configs/realtime_services.yaml",
    )

    PIPELINE = RealtimeDigitalHumanPipeline(cfg=cfg)

    VIDEO_TRACK = AvatarVideoTrack(
        fps=int(fps),
        width=int(width),
        height=int(height),
    )

    AUDIO_TRACK = AvatarAudioTrack(
        sample_rate=48000,
        channels=1,
        frame_duration_ms=20,
    )

    print("========== Runtime Ready ==========")
    print("device:", device)
    print("fps:", fps)
    print("video size:", width, height)


@app.on_event("startup")
async def on_startup():
    global INFER_LOCK
    INFER_LOCK = asyncio.Lock()


# ============================================================
# 页面
# ============================================================

@app.get("/")
async def index():
    return HTMLResponse(DEFAULT_HTML)


# ============================================================
# WebRTC Offer
# ============================================================

@app.post("/offer")
async def offer(req: OfferRequest):
    global VIDEO_TRACK, AUDIO_TRACK

    if VIDEO_TRACK is None or AUDIO_TRACK is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "tracks not initialized",
            },
            status_code=500,
        )

    pc = RTCPeerConnection()
    PCS.add(pc)

    # 同时添加 video + audio track
    pc.addTrack(VIDEO_TRACK)
    pc.addTrack(AUDIO_TRACK)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("[WebRTC] state:", pc.connectionState)

        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            PCS.discard(pc)

    offer_desc = RTCSessionDescription(
        sdp=req.sdp,
        type=req.type,
    )

    await pc.setRemoteDescription(offer_desc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


# ============================================================
# 加载数字人
# ============================================================

@app.post("/load_avatar")
async def load_avatar(
    avatar: UploadFile = File(...),
):
    global PIPELINE, VIDEO_TRACK, CURRENT_AVATAR_PATH

    if PIPELINE is None or VIDEO_TRACK is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "pipeline not initialized",
            },
            status_code=500,
        )

    suffix = Path(avatar.filename or "avatar.png").suffix
    if not suffix:
        suffix = ".png"

    save_path = UPLOAD_DIR / f"avatar_{uuid.uuid4().hex}{suffix}"

    async with aiofiles.open(save_path, "wb") as f:
        content = await avatar.read()
        await f.write(content)

    CURRENT_AVATAR_PATH = str(save_path)

    def _load():
        return PIPELINE.load_avatar(
            avatar_path=CURRENT_AVATAR_PATH,
            bbox_shift=PIPELINE.cfg.bbox_shift,
            extra_margin=PIPELINE.cfg.extra_margin,
        )

    idle_frame = await asyncio.to_thread(_load)

    VIDEO_TRACK.set_idle_frame(idle_frame)

    return {
        "ok": True,
        "message": "数字人已加载",
        "avatar_path": CURRENT_AVATAR_PATH,
    }


# ============================================================
# 生成本轮回答媒体
# ============================================================

def generate_llm_tts_avatar_media(user_text: str):
    """阻塞函数：LLM -> TTS -> MuseTalk video frames。

    返回：
        assistant_text, tts_audio_path, frames
    """

    if PIPELINE is None:
        raise RuntimeError("PIPELINE is None")

    if PIPELINE.avatar_session is None:
        raise RuntimeError("请先加载数字人")

    assistant_text, tts_audio_path = PIPELINE.llm_tts_audio(
        user_text=user_text,
        reference_audio=None,
    )

    frames: List = []

    for item in PIPELINE.avatar_service.stream_from_audio_file_with_session(
        avatar_session=PIPELINE.avatar_session,
        audio_path=tts_audio_path,
        fps=PIPELINE.cfg.fps,
        parsing_mode=PIPELINE.cfg.parsing_mode,
        realtime_sleep=False,
    ):
        frames.append(item.frame)

    return assistant_text, tts_audio_path, frames


def generate_tts_avatar_media(text: str):
    """阻塞函数：TTS -> MuseTalk video frames。"""

    if PIPELINE is None:
        raise RuntimeError("PIPELINE is None")

    if PIPELINE.avatar_session is None:
        raise RuntimeError("请先加载数字人")

    tts_audio_path = PIPELINE._tts_synthesize(text=text)

    frames: List = []

    for item in PIPELINE.avatar_service.stream_from_audio_file_with_session(
        avatar_session=PIPELINE.avatar_session,
        audio_path=tts_audio_path,
        fps=PIPELINE.cfg.fps,
        parsing_mode=PIPELINE.cfg.parsing_mode,
        realtime_sleep=False,
    ):
        frames.append(item.frame)

    return text, tts_audio_path, frames


async def play_media(
    frames: List,
    wav_path: str,
):
    """把视频帧和 wav 同时塞进 WebRTC Track。"""

    if VIDEO_TRACK is None or AUDIO_TRACK is None:
        raise RuntimeError("tracks not initialized")

    # 先清空旧队列，避免上一轮残留
    await VIDEO_TRACK.clear()
    await AUDIO_TRACK.clear()

    # 先把音频和视频都入队，再由 recv() 按时间消费
    await VIDEO_TRACK.enqueue_frames(frames)
    await AUDIO_TRACK.enqueue_wav(wav_path)


# ============================================================
# 文本问答：LLM + TTS + WebRTC AV
# ============================================================

@app.post("/ask")
async def ask(req: AskRequest):
    global INFER_LOCK

    if PIPELINE is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "pipeline not initialized",
            },
            status_code=500,
        )

    if PIPELINE.avatar_session is None:
        return {
            "ok": False,
            "message": "请先加载数字人",
        }

    user_text = (req.text or "").strip()

    if not user_text:
        return {
            "ok": False,
            "message": "问题为空",
        }

    async with INFER_LOCK:
        assistant_text, tts_audio_path, frames = await asyncio.to_thread(
            generate_llm_tts_avatar_media,
            user_text,
        )

        await play_media(
            frames=frames,
            wav_path=tts_audio_path,
        )

    return {
        "ok": True,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "tts_audio_path": tts_audio_path,
        "frames": len(frames),
        "message": "已通过 WebRTC 推送音视频",
    }


# ============================================================
# 直接播报：TTS + WebRTC AV
# ============================================================

@app.post("/speak")
async def speak(req: SpeakRequest):
    global INFER_LOCK

    if PIPELINE is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "pipeline not initialized",
            },
            status_code=500,
        )

    if PIPELINE.avatar_session is None:
        return {
            "ok": False,
            "message": "请先加载数字人",
        }

    text = (req.text or "").strip()

    if not text:
        return {
            "ok": False,
            "message": "文本为空",
        }

    async with INFER_LOCK:
        spoken_text, tts_audio_path, frames = await asyncio.to_thread(
            generate_tts_avatar_media,
            text,
        )

        await play_media(
            frames=frames,
            wav_path=tts_audio_path,
        )

    return {
        "ok": True,
        "text": spoken_text,
        "tts_audio_path": tts_audio_path,
        "frames": len(frames),
        "message": "已通过 WebRTC 推送音视频",
    }


# ============================================================
# 清空对话历史
# ============================================================

@app.post("/clear_history")
async def clear_history():
    if PIPELINE is not None:
        PIPELINE.clear_history()

    return {
        "ok": True,
        "message": "对话历史已清空",
    }


# ============================================================
# 关闭
# ============================================================

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in PCS]
    await asyncio.gather(*coros)
    PCS.clear()


# ============================================================
# 前端页面
# ============================================================

DEFAULT_HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>XumingAvatar WebRTC AV</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 24px;
            background: #111;
            color: #eee;
        }

        .container {
            display: flex;
            gap: 24px;
        }

        .left {
            width: 430px;
        }

        .right {
            flex: 1;
        }

        video {
            width: 864px;
            height: 1536px;
            background: #000;
            border-radius: 8px;
            border: 1px solid #444;
        }

        textarea {
            width: 100%;
            height: 100px;
            background: #222;
            color: #eee;
            border: 1px solid #555;
            padding: 8px;
            box-sizing: border-box;
        }

        button {
            margin-top: 8px;
            margin-right: 8px;
            padding: 8px 16px;
            cursor: pointer;
        }

        input {
            margin-top: 8px;
        }

        pre {
            background: #222;
            padding: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            min-height: 180px;
        }

        .tip {
            color: #aaa;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h2>XumingAvatar WebRTC 音视频数字人</h2>
    <p class="tip">
        当前版本：视频和音频都通过 WebRTC 传输。输入问题后，后端会先生成本轮音视频缓存，然后同步推送到浏览器。
    </p>

    <div class="container">
        <div class="left">
            <h3>1. WebRTC 连接</h3>
            <button onclick="startWebRTC()">连接 WebRTC</button>

            <h3>2. 加载数字人</h3>
            <input id="avatarFile" type="file" accept="image/*,video/*" />
            <br />
            <button onclick="loadAvatar()">加载数字人</button>

            <h3>3. 文本问答</h3>
            <textarea id="question">你好，我叫李雷，1+1等于多少？</textarea>
            <br />
            <button onclick="ask()">发送问题</button>
            <button onclick="clearHistory()">清空历史</button>

            <h3>4. 直接播报</h3>
            <textarea id="speakText">你好，我是实时数字人，现在音频和视频都已经通过 WebRTC 推送。</textarea>
            <br />
            <button onclick="speak()">直接播报</button>

            <h3>状态</h3>
            <pre id="log"></pre>
        </div>

        <div class="right">
            <h3>WebRTC 音视频流</h3>
            <video id="remoteVideo" autoplay playsinline controls></video>
            <p class="tip">
                如果没有声音，请点击视频控件的播放按钮，或检查浏览器是否拦截自动播放声音。
            </p>
        </div>
    </div>

<script>
let pc = null;
let remoteStream = null;

function log(msg) {
    const el = document.getElementById("log");
    el.textContent += msg + "\n";
    el.scrollTop = el.scrollHeight;
}

async function startWebRTC() {
    if (pc) {
        log("WebRTC 已经连接");
        return;
    }

    pc = new RTCPeerConnection();
    remoteStream = new MediaStream();

    const video = document.getElementById("remoteVideo");
    video.srcObject = remoteStream;

    pc.ontrack = function(event) {
        remoteStream.addTrack(event.track);
        video.srcObject = remoteStream;

        log("收到远端 track: " + event.track.kind);

        video.play().catch(err => {
            log("video play blocked: " + err);
        });
    };

    pc.onconnectionstatechange = function() {
        log("connectionState: " + pc.connectionState);
    };

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch("/offer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type
        })
    });

    const answer = await resp.json();

    await pc.setRemoteDescription(answer);

    log("WebRTC 已连接");
}

async function loadAvatar() {
    const fileInput = document.getElementById("avatarFile");

    if (!fileInput.files.length) {
        alert("请先选择人物图片或视频");
        return;
    }

    const formData = new FormData();
    formData.append("avatar", fileInput.files[0]);

    log("正在加载数字人...");

    const resp = await fetch("/load_avatar", {
        method: "POST",
        body: formData
    });

    const data = await resp.json();

    log(JSON.stringify(data, null, 2));
}

async function ask() {
    const text = document.getElementById("question").value;

    if (!text.trim()) {
        alert("请输入问题");
        return;
    }

    log("正在生成回答，请等待...");

    const resp = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    const data = await resp.json();

    log(JSON.stringify(data, null, 2));

    const video = document.getElementById("remoteVideo");
    video.play().catch(err => {
        log("play blocked: " + err);
    });
}

async function speak() {
    const text = document.getElementById("speakText").value;

    if (!text.trim()) {
        alert("请输入播报文本");
        return;
    }

    log("正在生成播报，请等待...");

    const resp = await fetch("/speak", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    const data = await resp.json();

    log(JSON.stringify(data, null, 2));

    const video = document.getElementById("remoteVideo");
    video.play().catch(err => {
        log("play blocked: " + err);
    });
}

async function clearHistory() {
    const resp = await fetch("/clear_history", {
        method: "POST"
    });

    const data = await resp.json();

    log(JSON.stringify(data, null, 2));
}
</script>
</body>
</html>
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--bbox_shift", default=0, type=int)
    parser.add_argument("--extra_margin", default=10, type=int)

    parser.add_argument(
        "--parsing_mode",
        default="jaw",
        choices=["jaw", "raw", "face"],
    )

    parser.add_argument("--width", default=864, type=int)
    parser.add_argument("--height", default=1536, type=int)

    args = parser.parse_args()

    init_runtime(
        device=args.device,
        fps=args.fps,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        width=args.width,
        height=args.height,
    )

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()