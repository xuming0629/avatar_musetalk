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
import re
import queue
import threading
from dataclasses import dataclass
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

# 第四版真正流式任务状态。
# STREAM_CANCEL_EVENT 用于新请求打断旧请求。
# STREAM_TASK 是后台 answer/speak 流式任务。
# FRAME_WORKER_LOCK 保证同一时间只有一个 MuseTalk 推理线程使用模型。
STREAM_CANCEL_EVENT: Optional[threading.Event] = None
STREAM_TASK: Optional[asyncio.Task] = None
FRAME_WORKER_LOCK = threading.Lock()

# v4 分句策略：
# - 不要一句太短就 TTS，否则 TTS/Whisper/MuseTalk 频繁重启，段间卡顿明显；
# - 默认至少累计 18 个中文字符；
# - 过长则强制切段，避免首包等待太久。
SENTENCE_END_RE = re.compile(r"[。！？!?；;\n]")
MAX_STREAM_SENTENCE_CHARS = 80
MIN_STREAM_SENTENCE_CHARS = 18

# v4 流水线参数：
# - TTS_QUEUE_MAX：允许 TTS 最多提前准备多少段音频；
# - AVATAR_PREBUFFER_FRAMES：每段音频播放前，先预生成几帧视频，减少音画开头卡顿；
# - TEXT_QUEUE_MAX：LLM 分句缓存长度。
TEXT_QUEUE_MAX = 32
TTS_QUEUE_MAX = 4
AVATAR_PREBUFFER_FRAMES = 6

QUEUE_TIMEOUT = 0.1


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
# 第四版：LLM / TTS / Avatar 三段流水线
# ============================================================

@dataclass
class TextSegment:
    index: int
    text: str


@dataclass
class AudioSegment:
    index: int
    text: str
    wav_path: str


class StreamEnd:
    pass


STREAM_END = StreamEnd()


def _append_future(coro, loop: asyncio.AbstractEventLoop):
    """在线程中安全调用 asyncio 协程，并等待入队完成。"""
    future = asyncio.run_coroutine_threadsafe(
        coro,
        loop,
    )
    return future.result()


def _safe_put(
    q: queue.Queue,
    item,
    cancel_event: threading.Event,
):
    """可取消的 queue.put，避免 worker 在退出时卡死。"""
    while not cancel_event.is_set():
        try:
            q.put(
                item,
                timeout=QUEUE_TIMEOUT,
            )
            return True
        except queue.Full:
            continue
    return False


def _safe_get(
    q: queue.Queue,
    cancel_event: threading.Event,
):
    """可取消的 queue.get。"""
    while not cancel_event.is_set():
        try:
            return q.get(
                timeout=QUEUE_TIMEOUT,
            )
        except queue.Empty:
            continue
    return STREAM_END


def _pop_stream_sentence(buffer: str, force: bool = False):
    """从 token 缓冲区里切出一个适合 TTS 的段落。

    v4 策略：
        1. 优先按句末标点切；
        2. 如果句子太短，不立即切，继续等下一句合并；
        3. 超过 MAX_STREAM_SENTENCE_CHARS 强制切，避免首包太慢；
        4. force=True 时把剩余内容全部切出。
    """
    buffer = buffer or ""

    if not buffer.strip():
        return None, ""

    # 按句末标点切分，但太短的句子先缓存，合并下一句。
    for idx, ch in enumerate(buffer):
        if SENTENCE_END_RE.match(ch):
            sentence = buffer[:idx + 1].strip()
            remain = buffer[idx + 1:]

            if len(sentence) >= MIN_STREAM_SENTENCE_CHARS or force:
                return sentence, remain

            # 太短则不切，等待后续 token 继续合并。
            break

    # 文本过长时强制切分，避免等待太久。
    if len(buffer) >= MAX_STREAM_SENTENCE_CHARS:
        cut = MAX_STREAM_SENTENCE_CHARS

        for mark in ["，", ",", "、", " "]:
            pos = buffer.rfind(
                mark,
                0,
                MAX_STREAM_SENTENCE_CHARS,
            )

            if pos >= MIN_STREAM_SENTENCE_CHARS:
                cut = pos + 1
                break

        return buffer[:cut].strip(), buffer[cut:]

    if force:
        sentence = buffer.strip()
        return sentence, ""

    return None, buffer


def _synthesize_sentence(sentence: str) -> str:
    """同步 TTS：短句 -> wav。"""
    if PIPELINE is None:
        raise RuntimeError("PIPELINE is None")

    sentence = (sentence or "").strip()

    if not sentence:
        raise RuntimeError("TTS sentence is empty")

    return PIPELINE.tts_synthesize(
        text=sentence,
        reference_audio=None,
    )


def _llm_text_producer_worker(
    user_text: str,
    text_queue: queue.Queue,
    cancel_event: threading.Event,
    shared_state: dict,
):
    """LLM token 流式输出，切成 TextSegment 放入 text_queue。"""
    if PIPELINE is None:
        raise RuntimeError("PIPELINE is None")

    final_parts: List[str] = []
    buffer = ""
    seg_idx = 0

    print("[StreamV4][LLM] start:", user_text)

    try:
        for token in PIPELINE.llm_service.stream_chat(
            user_text,
            history=PIPELINE.history,
        ):
            if cancel_event.is_set():
                break

            token = str(token or "")

            if not token:
                continue

            final_parts.append(token)
            buffer += token

            while True:
                sentence, buffer = _pop_stream_sentence(
                    buffer,
                    force=False,
                )

                if not sentence:
                    break

                seg = TextSegment(
                    index=seg_idx,
                    text=sentence,
                )
                seg_idx += 1

                print("[StreamV4][TEXT]", seg.index, seg.text)

                if not _safe_put(
                    text_queue,
                    seg,
                    cancel_event,
                ):
                    break

        sentence, buffer = _pop_stream_sentence(
            buffer,
            force=True,
        )

        if sentence and not cancel_event.is_set():
            seg = TextSegment(
                index=seg_idx,
                text=sentence,
            )
            seg_idx += 1

            print("[StreamV4][TEXT]", seg.index, seg.text)

            _safe_put(
                text_queue,
                seg,
                cancel_event,
            )

        shared_state["answer"] = "".join(final_parts).strip()

    except Exception as e:
        shared_state["error"] = repr(e)
        print("[StreamV4][ERROR] LLM worker failed:", repr(e))

    finally:
        _safe_put(
            text_queue,
            STREAM_END,
            cancel_event,
        )


def _plain_text_producer_worker(
    text: str,
    text_queue: queue.Queue,
    cancel_event: threading.Event,
    shared_state: dict,
):
    """直接播报文本切段，放入 text_queue。"""
    buffer = text or ""
    seg_idx = 0
    final_parts: List[str] = []

    try:
        while True:
            sentence, buffer = _pop_stream_sentence(
                buffer,
                force=False,
            )

            if not sentence:
                break

            seg = TextSegment(
                index=seg_idx,
                text=sentence,
            )
            seg_idx += 1
            final_parts.append(sentence)

            print("[StreamV4][TEXT]", seg.index, seg.text)

            if not _safe_put(
                text_queue,
                seg,
                cancel_event,
            ):
                break

        sentence, _ = _pop_stream_sentence(
            buffer,
            force=True,
        )

        if sentence and not cancel_event.is_set():
            seg = TextSegment(
                index=seg_idx,
                text=sentence,
            )
            final_parts.append(sentence)

            print("[StreamV4][TEXT]", seg.index, seg.text)

            _safe_put(
                text_queue,
                seg,
                cancel_event,
            )

        shared_state["answer"] = "".join(final_parts).strip()

    except Exception as e:
        shared_state["error"] = repr(e)
        print("[StreamV4][ERROR] text worker failed:", repr(e))

    finally:
        _safe_put(
            text_queue,
            STREAM_END,
            cancel_event,
        )


def _tts_worker(
    text_queue: queue.Queue,
    audio_queue: queue.Queue,
    cancel_event: threading.Event,
    shared_state: dict,
):
    """TTS worker：从 text_queue 取文本，提前合成 wav，放入 audio_queue。

    v4 关键点：
        这个 worker 和 Avatar worker 并行运行。
        Avatar 正在播放当前段时，TTS 可以提前生成下一段 wav。
    """
    total = 0

    try:
        while not cancel_event.is_set():
            item = _safe_get(
                text_queue,
                cancel_event,
            )

            if item is STREAM_END:
                break

            if not isinstance(item, TextSegment):
                continue

            text = (item.text or "").strip()

            if not text:
                continue

            print("[StreamV4][TTS:start]", item.index, text)

            wav_path = _synthesize_sentence(text)

            print("[StreamV4][TTS:done]", item.index, wav_path)

            audio_seg = AudioSegment(
                index=item.index,
                text=text,
                wav_path=wav_path,
            )

            if not _safe_put(
                audio_queue,
                audio_seg,
                cancel_event,
            ):
                break

            total += 1

    except Exception as e:
        shared_state["error"] = repr(e)
        print("[StreamV4][ERROR] TTS worker failed:", repr(e))

    finally:
        _safe_put(
            audio_queue,
            STREAM_END,
            cancel_event,
        )

        print("[StreamV4][TTS] finished, segments=", total)


def _enqueue_wav_for_segment(
    wav_path: str,
    loop: asyncio.AbstractEventLoop,
):
    """把当前段 wav 追加到 WebRTC AudioTrack。"""
    if AUDIO_TRACK is None:
        raise RuntimeError("AUDIO_TRACK is None")

    _append_future(
        AUDIO_TRACK.enqueue_wav_append(wav_path),
        loop,
    )


def _enqueue_video_frame(
    frame,
    loop: asyncio.AbstractEventLoop,
):
    """把当前帧追加到 WebRTC VideoTrack。"""
    if VIDEO_TRACK is None:
        raise RuntimeError("VIDEO_TRACK is None")

    _append_future(
        VIDEO_TRACK.enqueue_frame(frame),
        loop,
    )


def _avatar_worker(
    audio_queue: queue.Queue,
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
    shared_state: dict,
):
    """Avatar worker：wav -> audio chunks -> MuseTalk frame -> WebRTC。

    v4 优化：
        1. TTS 已经提前准备好 wav；
        2. 每段先预生成 AVATAR_PREBUFFER_FRAMES 帧；
        3. 预缓冲完成后，音频和视频一起入队；
        4. 后续帧边生成边入队。
    """
    if PIPELINE is None:
        raise RuntimeError("PIPELINE is None")

    if PIPELINE.avatar_session is None:
        raise RuntimeError("请先加载数字人")

    total_frames = 0
    total_segments = 0

    try:
        while not cancel_event.is_set():
            item = _safe_get(
                audio_queue,
                cancel_event,
            )

            if item is STREAM_END:
                break

            if not isinstance(item, AudioSegment):
                continue

            print(
                "[StreamV4][AVATAR:start]",
                item.index,
                item.text,
            )

            segment_frames = 0
            prebuffer_frames = []

            # 保护底层模型：同一时间只允许一条 MuseTalk 推理链路运行。
            with FRAME_WORKER_LOCK:
                frame_iter = PIPELINE.avatar_service.stream_from_audio_file_with_session(
                    avatar_session=PIPELINE.avatar_session,
                    audio_path=item.wav_path,
                    fps=PIPELINE.cfg.fps,
                    parsing_mode=PIPELINE.cfg.parsing_mode,
                    realtime_sleep=False,
                )

                # 先预生成几帧，但暂时不推音频，避免视频还没准备好音频先播放。
                for stream_frame in frame_iter:
                    if cancel_event.is_set():
                        break

                    prebuffer_frames.append(stream_frame.frame)
                    segment_frames += 1

                    if len(prebuffer_frames) >= AVATAR_PREBUFFER_FRAMES:
                        break

                if cancel_event.is_set():
                    break

                # 先把预缓冲视频帧入队，然后立刻追加音频。
                for frame in prebuffer_frames:
                    _enqueue_video_frame(
                        frame,
                        loop,
                    )

                _enqueue_wav_for_segment(
                    item.wav_path,
                    loop,
                )

                # 剩余帧边生成边推。
                for stream_frame in frame_iter:
                    if cancel_event.is_set():
                        break

                    _enqueue_video_frame(
                        stream_frame.frame,
                        loop,
                    )

                    segment_frames += 1

            total_segments += 1
            total_frames += segment_frames

            print(
                "[StreamV4][AVATAR:done]",
                item.index,
                "frames=",
                segment_frames,
            )

    except Exception as e:
        shared_state["error"] = repr(e)
        print("[StreamV4][ERROR] Avatar worker failed:", repr(e))

    finally:
        shared_state["frames"] = total_frames
        shared_state["segments"] = total_segments

        print(
            "[StreamV4][AVATAR] finished, segments=",
            total_segments,
            "frames=",
            total_frames,
        )


def _run_pipeline_stream_workers(
    user_text: Optional[str],
    speak_text: Optional[str],
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
):
    """v4 流水线总控。

    三个队列：
        text_queue:
            LLM/text 分句结果。

        audio_queue:
            TTS 已经合成好的 wav 段。

        WebRTC queue:
            AUDIO_TRACK / VIDEO_TRACK 内部队列。

    三类 worker：
        1. text producer:
            LLM token -> 分句，或者直接文本 -> 分句。

        2. tts worker:
            文本段 -> wav。
            可以在 Avatar 播当前段时提前准备下一段。

        3. avatar worker:
            wav -> MuseTalk frames -> WebRTC。
    """
    if PIPELINE is None:
        raise RuntimeError("PIPELINE is None")

    text_queue: queue.Queue = queue.Queue(
        maxsize=TEXT_QUEUE_MAX,
    )

    audio_queue: queue.Queue = queue.Queue(
        maxsize=TTS_QUEUE_MAX,
    )

    shared_state = {
        "answer": "",
        "frames": 0,
        "segments": 0,
        "error": None,
    }

    if user_text is not None:
        producer = threading.Thread(
            target=_llm_text_producer_worker,
            args=(
                user_text,
                text_queue,
                cancel_event,
                shared_state,
            ),
            daemon=True,
            name="stream-v4-llm-worker",
        )
    else:
        producer = threading.Thread(
            target=_plain_text_producer_worker,
            args=(
                speak_text or "",
                text_queue,
                cancel_event,
                shared_state,
            ),
            daemon=True,
            name="stream-v4-text-worker",
        )

    tts_thread = threading.Thread(
        target=_tts_worker,
        args=(
            text_queue,
            audio_queue,
            cancel_event,
            shared_state,
        ),
        daemon=True,
        name="stream-v4-tts-worker",
    )

    avatar_thread = threading.Thread(
        target=_avatar_worker,
        args=(
            audio_queue,
            loop,
            cancel_event,
            shared_state,
        ),
        daemon=True,
        name="stream-v4-avatar-worker",
    )

    producer.start()
    tts_thread.start()
    avatar_thread.start()

    producer.join()
    tts_thread.join()
    avatar_thread.join()

    final_text = (shared_state.get("answer") or "").strip()

    if (
        user_text is not None
        and final_text
        and not cancel_event.is_set()
        and shared_state.get("error") is None
    ):
        PIPELINE.history.append({
            "role": "user",
            "content": user_text,
        })
        PIPELINE.history.append({
            "role": "assistant",
            "content": final_text,
        })

    if shared_state.get("error"):
        print("[StreamV4][ERROR] pipeline error:", shared_state["error"])

    print(
        "[StreamV4] finished, cancelled=",
        cancel_event.is_set(),
        "segments=",
        shared_state.get("segments"),
        "frames=",
        shared_state.get("frames"),
        "answer=",
        final_text,
    )


async def _cancel_previous_stream():
    """新请求到来时，通知上一轮停止。"""
    global STREAM_CANCEL_EVENT, STREAM_TASK

    if STREAM_CANCEL_EVENT is not None:
        STREAM_CANCEL_EVENT.set()

    STREAM_CANCEL_EVENT = None
    STREAM_TASK = None


async def _prepare_tracks_for_new_stream():
    """清理上一轮音视频队列。"""
    if VIDEO_TRACK is None or AUDIO_TRACK is None:
        raise RuntimeError("tracks not initialized")

    await VIDEO_TRACK.clear()
    await AUDIO_TRACK.clear()


async def start_llm_tts_avatar_stream(user_text: str):
    """启动一轮 LLM + TTS + 数字人 v4 流水线回答。"""
    global STREAM_CANCEL_EVENT, STREAM_TASK

    await _cancel_previous_stream()
    await _prepare_tracks_for_new_stream()

    loop = asyncio.get_running_loop()
    STREAM_CANCEL_EVENT = threading.Event()

    STREAM_TASK = asyncio.create_task(
        asyncio.to_thread(
            _run_pipeline_stream_workers,
            user_text,
            None,
            loop,
            STREAM_CANCEL_EVENT,
        )
    )

    return STREAM_TASK


async def start_tts_avatar_stream(text: str):
    """启动一轮直接播报 v4 流水线。"""
    global STREAM_CANCEL_EVENT, STREAM_TASK

    await _cancel_previous_stream()
    await _prepare_tracks_for_new_stream()

    loop = asyncio.get_running_loop()
    STREAM_CANCEL_EVENT = threading.Event()

    STREAM_TASK = asyncio.create_task(
        asyncio.to_thread(
            _run_pipeline_stream_workers,
            None,
            text,
            loop,
            STREAM_CANCEL_EVENT,
        )
    )

    return STREAM_TASK



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

    # 真流式版本：这里只启动后台流式任务，HTTP 立即返回。
    # 后台任务会：LLM token -> 分句 TTS -> 数字人逐帧 WebRTC 推送。
    async with INFER_LOCK:
        await start_llm_tts_avatar_stream(user_text)

    return {
        "ok": True,
        "streaming": True,
        "user_text": user_text,
        "message": "已开始 LLM + TTS + 数字人 v4 流水线实时回答",
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
        await start_tts_avatar_stream(text)

    return {
        "ok": True,
        "streaming": True,
        "text": text,
        "message": "已开始 TTS + 数字人 v4 流水线实时播报",
    }


@app.post("/stop")
async def stop_stream():
    await _cancel_previous_stream()

    if VIDEO_TRACK is not None:
        await VIDEO_TRACK.clear()

    if AUDIO_TRACK is not None:
        await AUDIO_TRACK.clear()

    return {
        "ok": True,
        "message": "已停止当前流式回答",
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
        当前版本：v4 流水线实时回答。LLM 流式分句，TTS 提前合成下一段，数字人按帧通过 WebRTC 推送。
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
            <button onclick="stopStream()">停止回答</button>

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

async function stopStream() {
    const resp = await fetch("/stop", {
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