#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from services.asr.base import (
    DummyASRService,
    MuseTalkWhisperASRService,
)


def _get_str(cfg: Dict[str, Any], key: str, default: str = "") -> str:
    value = cfg.get(key, default)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def build_asr_service(
    cfg: Dict[str, Any],
    audio_processor=None,
):
    asr_cfg = cfg.get("asr", {})
    provider = _get_str(asr_cfg, "provider", "musetalk_whisper").lower()

    if provider == "dummy":
        dummy_cfg = asr_cfg.get("dummy", {})
        return DummyASRService(
            fixed_text=_get_str(
                dummy_cfg,
                "fixed_text",
                "你好，我是实时数字人。",
            )
        )

    if provider in ("musetalk_whisper", "whisper", "audio2feature"):
        whisper_cfg = asr_cfg.get("musetalk_whisper", {})

        return MuseTalkWhisperASRService(
            audio_processor=audio_processor,
            config_path=_get_str(
                whisper_cfg,
                "config_path",
                "configs/musetalk_v15.yaml",
            ),
            project_root=_get_str(
                whisper_cfg,
                "project_root",
                "",
            ) or None,
            device=_get_str(
                whisper_cfg,
                "device",
                "",
            ) or None,
        )

    raise ValueError(f"不支持的 ASR provider: {provider}")