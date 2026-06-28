#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Optional

import yaml

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

    当前实现：
    1. 音频文件驱动数字人
    2. 文本 TTS 播报数字人
    3. 用户文本 -> LLM -> TTS -> 数字人
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

        # MuseTalk 实时服务
        self.avatar_service = RealtimeMuseTalkAvatarService(cfg=self.cfg)

        # TTS 服务
        self.tts_service = build_tts_service(self.service_cfg)

        # LLM 服务
        self.llm_service = build_llm_service(self.service_cfg)

        # 多轮对话历史
        self.history: List[Dict[str, str]] = []

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
        """音频文件驱动数字人。"""

        yield from self.avatar_service.stream_from_audio_file(
            avatar_path=avatar_path,
            audio_path=audio_path,
            fps=fps,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            realtime_sleep=realtime_sleep,
        )

    def stream_text(
        self,
        avatar_path: str,
        text: str,
        reference_audio: Optional[str] = None,
        fps: Optional[int] = None,
        bbox_shift: Optional[int] = None,
        extra_margin: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        realtime_sleep: bool = True,
    ) -> Generator[StreamFrame, None, None]:
        """文本 -> TTS -> 数字人。"""

        text = (text or "").strip()

        if not text:
            raise RuntimeError("文本为空，无法生成 TTS 音频")

        tts_audio_path = self._tts_synthesize(
            text=text,
            reference_audio=reference_audio,
        )

        print("[TTS] audio:", tts_audio_path)

        yield from self.stream_audio_file(
            avatar_path=avatar_path,
            audio_path=tts_audio_path,
            fps=fps,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            realtime_sleep=realtime_sleep,
        )

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

        self.history.append({
            "role": "user",
            "content": user_text,
        })

        self.history.append({
            "role": "assistant",
            "content": assistant_text,
        })

        return assistant_text

    def stream_llm_text(
        self,
        avatar_path: str,
        user_text: str,
        reference_audio: Optional[str] = None,
        fps: Optional[int] = None,
        bbox_shift: Optional[int] = None,
        extra_margin: Optional[int] = None,
        parsing_mode: Optional[str] = None,
        realtime_sleep: bool = True,
    ) -> Generator[StreamFrame, None, None]:
        """用户文本 -> LLM -> TTS -> 数字人。"""

        assistant_text = self.run_llm_text(user_text)

        print("[LLM]", assistant_text)

        yield from self.stream_text(
            avatar_path=avatar_path,
            text=assistant_text,
            reference_audio=reference_audio,
            fps=fps,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            realtime_sleep=realtime_sleep,
        )

    def llm_tts_audio(
        self,
        user_text: str,
        reference_audio: Optional[str] = None,
    ):
        """用户文本 -> LLM -> TTS，仅返回文本和音频路径，方便 Gradio 显示中间结果。"""

        assistant_text = self.run_llm_text(user_text)

        tts_audio_path = self._tts_synthesize(
            text=assistant_text,
            reference_audio=reference_audio,
        )

        return assistant_text, tts_audio_path

    def clear_history(self):
        self.history.clear()

    def _tts_synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
    ) -> str:
        """兼容不同 TTS service 的 synthesize 接口。"""

        if not hasattr(self.tts_service, "synthesize"):
            raise RuntimeError("当前 TTS service 没有 synthesize 方法")

        # synthesize(text, reference_audio=xxx)
        try:
            return self.tts_service.synthesize(
                text,
                reference_audio=reference_audio,
            )
        except TypeError:
            pass

        # synthesize(text)
        try:
            return self.tts_service.synthesize(text)
        except TypeError:
            pass

        # synthesize(text=text)
        try:
            return self.tts_service.synthesize(text=text)
        except TypeError as e:
            raise RuntimeError("TTS synthesize 调用失败，请检查 services/tts/base.py") from e