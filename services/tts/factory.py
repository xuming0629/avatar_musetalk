#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

from services.tts.base import (
    DummyTTSService,
    EdgeTTSService,
)


def build_tts_service(cfg: Dict[str, Any]):
    tts_cfg = cfg.get("tts", {})
    provider = str(tts_cfg.get("provider", "dummy")).strip().lower()

    if provider == "dummy":
        dummy_cfg = tts_cfg.get("dummy", {})
        return DummyTTSService(
            out_dir=str(dummy_cfg.get("out_dir", "./results/tts")),
            seconds=float(dummy_cfg.get("seconds", 3.0)),
            sample_rate=int(dummy_cfg.get("sample_rate", 16000)),
        )

    if provider in ("edge", "edgetts", "edge_tts"):
        edge_cfg = tts_cfg.get("edgetts", {})
        return EdgeTTSService(
            out_dir=str(edge_cfg.get("out_dir", "./results/tts")),
            voice=str(edge_cfg.get("voice", "zh-CN-XiaoxiaoNeural")),
            rate=str(edge_cfg.get("rate", "+0%")),
            volume=str(edge_cfg.get("volume", "+0%")),
            pitch=str(edge_cfg.get("pitch", "+0Hz")),
            sample_rate=int(edge_cfg.get("sample_rate", 16000)),
        )

    raise ValueError(f"不支持的 TTS provider: {provider}")