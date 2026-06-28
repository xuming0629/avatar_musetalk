#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional


class BaseASRService:
    """ASR 服务接口。

    输入：音频文件路径
    输出：识别文本
    """

    name = "base"

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class DummyASRService(BaseASRService):
    """占位 ASR：不真正识别，只返回固定文本。"""

    name = "dummy"

    def __init__(self, fixed_text: str = "你好，我是实时数字人。"):
        self.fixed_text = fixed_text

    def transcribe(self, audio_path: str) -> str:
        return self.fixed_text


class MuseTalkWhisperASRService(BaseASRService):
    """复用 MuseTalk Audio2Feature 里的 Whisper 模型做 ASR。

    好处：
    - 不需要额外安装 openai-whisper
    - 不需要额外加载一个 Whisper 模型
    - 和你原来离线数字人的 Whisper 模型保持一致
    """

    name = "musetalk_whisper"

    def __init__(
        self,
        audio_processor=None,
        config_path: str = "configs/musetalk_v15.yaml",
        project_root: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if audio_processor is not None:
            self.audio_processor = audio_processor
        else:
            from src.nets.whisper.audio2feature import Audio2Feature

            self.audio_processor = Audio2Feature(
                config_path=config_path,
                project_root=project_root,
                device=device,
            )

    def transcribe(self, audio_path: str) -> str:
        audio_path = str(audio_path)

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 优先使用你刚才加到 Audio2Feature 里的方法
        if hasattr(self.audio_processor, "transcribe_text"):
            text = self.audio_processor.transcribe_text(audio_path)
            return (text or "").strip()

        # 如果你暂时没改 Audio2Feature，就走 fallback
        from src.nets.whisper.transcribe import transcribe

        result = transcribe(
            self.audio_processor.model,
            audio_path,
        )

        segments = result.get("segments", [])
        texts = []

        for seg in segments:
            txt = seg.get("text", "")
            if txt:
                texts.append(str(txt).strip())

        return "".join(texts).strip()


# 为了兼容之前的名字，保留这些类，但不实现
class FunASRService(BaseASRService):
    name = "funasr"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "当前建议先使用 MuseTalkWhisperASRService，"
            "如果后续要接 FunASR，再单独实现。"
        )


class SenseVoiceASRService(BaseASRService):
    name = "sensevoice"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "当前建议先使用 MuseTalkWhisperASRService，"
            "如果后续要接 SenseVoice，再单独实现。"
        )