#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import yaml
import numpy as np

from services.types import RealtimeAvatarConfig, StreamFrame
from services.realtime_avatar import RealtimeMuseTalkAvatarService
from services.tts import build_tts_service
from services.llm import build_llm_service


def load_yaml_config(config_path: str) -> Dict:
    path = Path(config_path)

    if not path.exists():
        print(f"[Pipeline] config 不存在，使用空配置: {config_path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


class RealtimeDigitalHumanPipeline:
    """实时数字人 Pipeline。

    当前支持：

    1. 加载数字人：
        avatar_path -> AvatarSession -> idle_frame

    2. 文本问答：
        user_text -> LLM -> assistant_text

    3. 语音合成：
        assistant_text -> TTS -> tts_audio_path

    4. 数字人回答：
        tts_audio_path + 已缓存 AvatarSession -> stream frames

    5. WebRTC 模式：
        一次性生成本轮回答的：
        assistant_text, tts_audio_path, frames
    """

    def __init__(
        self,
        cfg: Optional[RealtimeAvatarConfig] = None,
        service_cfg: Optional[Dict] = None,
    ):
        self.cfg = cfg or RealtimeAvatarConfig()

        if service_cfg is None:
            service_cfg = load_yaml_config(self.cfg.config_path)

        self.service_cfg = service_cfg

        self.avatar_service = RealtimeMuseTalkAvatarService(cfg=self.cfg)
        self.tts_service = build_tts_service(self.service_cfg)
        self.llm_service = build_llm_service(self.service_cfg)

        self.history: List[Dict[str, str]] = []

        self.avatar_session = None
        self.avatar_path: Optional[str] = None
        self.idle_frame: Optional[np.ndarray] = None

    # ============================================================
    # 1. 加载数字人
    # ============================================================

    def load_avatar(
        self,
        avatar_path: str,
        bbox_shift: Optional[int] = None,
        extra_margin: Optional[int] = None,
    ) -> np.ndarray:
        """预加载数字人。

        只做一次：
        - 读取人物图片/视频
        - 人脸检测
        - VAE latent encode
        - 缓存 AvatarSession

        返回：
            idle_frame: np.ndarray, RGB
        """

        if not avatar_path or not Path(avatar_path).exists():
            raise RuntimeError(f"人物图片/视频不存在: {avatar_path}")

        self.avatar_session = self.avatar_service.build_avatar_session(
            avatar_path=avatar_path,
            bbox_shift=self.cfg.bbox_shift if bbox_shift is None else int(bbox_shift),
            extra_margin=self.cfg.extra_margin if extra_margin is None else int(extra_margin),
        )

        self.avatar_path = avatar_path

        idle_frame, _, _ = self.avatar_session.next()
        self.idle_frame = idle_frame

        print("[Pipeline] avatar loaded:", avatar_path)

        return idle_frame

    def has_loaded_avatar(self) -> bool:
        return self.avatar_session is not None

    def get_idle_frame(self) -> Optional[np.ndarray]:
        return self.idle_frame

    # ============================================================
    # 2. 旧模式：音频文件驱动
    # ============================================================

    def stream_audio_file(
        self,
        avatar_path: str,
        audio_path: str,
        fps: Optional[int] = None,
        bbox_shift: Optional[int] = None,
        extra_margin: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        realtime_sleep: bool = True,
    ) -> Generator[StreamFrame, None, None]:
        """旧模式：每次根据 avatar_path 重新 build session。"""

        yield from self.avatar_service.stream_from_audio_file(
            avatar_path=avatar_path,
            audio_path=audio_path,
            fps=fps,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            realtime_sleep=realtime_sleep,
        )

    # ============================================================
    # 3. 已加载数字人 + 音频文件驱动
    # ============================================================

    def stream_audio_with_loaded_avatar(
        self,
        audio_path: str,
        fps: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        realtime_sleep: bool = True,
    ) -> Generator[StreamFrame, None, None]:
        """复用已加载 AvatarSession，用音频文件驱动数字人。"""

        if self.avatar_session is None:
            raise RuntimeError("请先调用 load_avatar() 加载数字人")

        if not audio_path or not Path(audio_path).exists():
            raise RuntimeError(f"音频文件不存在: {audio_path}")

        yield from self.avatar_service.stream_from_audio_file_with_session(
            avatar_session=self.avatar_session,
            audio_path=audio_path,
            fps=fps or self.cfg.fps,
            parsing_mode=parsing_mode or self.cfg.parsing_mode,
            realtime_sleep=realtime_sleep,
        )

    # ============================================================
    # 4. 文本 -> TTS -> 已加载数字人
    # ============================================================

    def stream_text_with_loaded_avatar(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        fps: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        realtime_sleep: bool = True,
    ) -> Generator[Tuple[str, str, np.ndarray], None, None]:
        """文本直接 TTS 播报，不经过 LLM。

        返回：
            text, tts_audio_path, frame
        """

        if self.avatar_session is None:
            raise RuntimeError("请先调用 load_avatar() 加载数字人")

        text = (text or "").strip()

        if not text:
            raise RuntimeError("文本为空，无法播报")

        tts_audio_path = self.tts_synthesize(
            text=text,
            reference_audio=reference_audio,
        )

        for item in self.stream_audio_with_loaded_avatar(
            audio_path=tts_audio_path,
            fps=fps or self.cfg.fps,
            parsing_mode=parsing_mode or self.cfg.parsing_mode,
            realtime_sleep=realtime_sleep,
        ):
            yield text, tts_audio_path, item.frame

    # ============================================================
    # 5. 用户文本 -> LLM
    # ============================================================

    def run_llm_text(
        self,
        user_text: str,
    ) -> str:
        """用户文本 -> LLM 回复文本。"""

        user_text = (user_text or "").strip()

        if not user_text:
            return "你好，请问有什么可以帮你？"

        assistant_text = self.llm_service.chat(
            user_text,
            history=self.history,
        )

        assistant_text = (assistant_text or "").strip()

        if not assistant_text:
            assistant_text = "我暂时没有想到合适的回答。"

        self.history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        self.history.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

        return assistant_text

    # ============================================================
    # 6. 用户文本 -> LLM -> TTS
    # ============================================================

    def llm_tts_audio(
        self,
        user_text: str,
        reference_audio: Optional[str] = None,
    ) -> Tuple[str, str]:
        """用户文本 -> LLM -> TTS。

        返回：
            assistant_text, tts_audio_path
        """

        assistant_text = self.run_llm_text(user_text)

        tts_audio_path = self.tts_synthesize(
            text=assistant_text,
            reference_audio=reference_audio,
        )

        return assistant_text, tts_audio_path

    # ============================================================
    # 7. 用户文本 -> LLM -> TTS -> 已加载数字人流式回答
    # ============================================================

    def answer_with_loaded_avatar(
        self,
        user_text: str,
        reference_audio: Optional[str] = None,
        fps: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        realtime_sleep: bool = True,
    ) -> Generator[Tuple[str, str, np.ndarray], None, None]:
        """用户输入问题，已加载数字人回答。

        返回：
            assistant_text, tts_audio_path, frame
        """

        if self.avatar_session is None:
            raise RuntimeError("请先调用 load_avatar() 加载数字人")

        assistant_text, tts_audio_path = self.llm_tts_audio(
            user_text=user_text,
            reference_audio=reference_audio,
        )

        print("[LLM]", assistant_text)
        print("[TTS]", tts_audio_path)

        for item in self.stream_audio_with_loaded_avatar(
            audio_path=tts_audio_path,
            fps=fps or self.cfg.fps,
            parsing_mode=parsing_mode or self.cfg.parsing_mode,
            realtime_sleep=realtime_sleep,
        ):
            yield assistant_text, tts_audio_path, item.frame

    # ============================================================
    # 8. WebRTC 专用：LLM + TTS + 生成一轮视频帧
    # ============================================================

    def generate_llm_tts_avatar_media(
        self,
        user_text: str,
        reference_audio: Optional[str] = None,
        fps: Optional[int] = None,
        parsing_mode: Optional[str] = None,
    ) -> Tuple[str, str, List[np.ndarray]]:
        """WebRTC 专用。

        用户文本 -> LLM -> TTS -> MuseTalk frames

        返回：
            assistant_text
            tts_audio_path
            frames

        说明：
            这个方法是阻塞式的。
            适合在 FastAPI 里用 asyncio.to_thread() 调用。
        """

        if self.avatar_session is None:
            raise RuntimeError("请先调用 load_avatar() 加载数字人")

        assistant_text, tts_audio_path = self.llm_tts_audio(
            user_text=user_text,
            reference_audio=reference_audio,
        )

        frames: List[np.ndarray] = []

        for item in self.stream_audio_with_loaded_avatar(
            audio_path=tts_audio_path,
            fps=fps or self.cfg.fps,
            parsing_mode=parsing_mode or self.cfg.parsing_mode,
            realtime_sleep=False,
        ):
            frames.append(item.frame)

        return assistant_text, tts_audio_path, frames

    # ============================================================
    # 9. WebRTC 专用：直接 TTS + 生成一轮视频帧
    # ============================================================

    def generate_tts_avatar_media(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        fps: Optional[int] = None,
        parsing_mode: Optional[str] = None,
    ) -> Tuple[str, str, List[np.ndarray]]:
        """WebRTC 专用。

        文本 -> TTS -> MuseTalk frames

        返回：
            spoken_text
            tts_audio_path
            frames
        """

        if self.avatar_session is None:
            raise RuntimeError("请先调用 load_avatar() 加载数字人")

        text = (text or "").strip()

        if not text:
            raise RuntimeError("文本为空，无法播报")

        tts_audio_path = self.tts_synthesize(
            text=text,
            reference_audio=reference_audio,
        )

        frames: List[np.ndarray] = []

        for item in self.stream_audio_with_loaded_avatar(
            audio_path=tts_audio_path,
            fps=fps or self.cfg.fps,
            parsing_mode=parsing_mode or self.cfg.parsing_mode,
            realtime_sleep=False,
        ):
            frames.append(item.frame)

        return text, tts_audio_path, frames

    # ============================================================
    # 10. 工具
    # ============================================================

    def clear_history(self):
        self.history.clear()

    def tts_synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        """公开 TTS 合成接口。

        避免外部 WebRTC 服务端直接调用 _tts_synthesize。
        """

        return self._tts_synthesize(
            text=text,
            reference_audio=reference_audio,
        )

    def _tts_synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        """兼容不同 TTS service 的 synthesize 接口。"""

        if not hasattr(self.tts_service, "synthesize"):
            raise RuntimeError("当前 TTS service 没有 synthesize 方法")

        text = (text or "").strip()

        if not text:
            raise RuntimeError("TTS 文本为空")

        try:
            return self.tts_service.synthesize(
                text,
                reference_audio=reference_audio,
            )
        except TypeError:
            pass

        try:
            return self.tts_service.synthesize(text)
        except TypeError:
            pass

        try:
            return self.tts_service.synthesize(text=text)
        except TypeError as e:
            raise RuntimeError(
                "TTS synthesize 调用失败，请检查 services/tts/base.py"
            ) from e