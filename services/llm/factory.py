#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict

from services.llm.base import EchoLLMService, KimiLLMService


def build_llm_service(cfg: Dict[str, Any]):
    """根据配置创建 LLM 服务。"""

    llm_cfg = cfg.get("llm", {})
    provider = str(llm_cfg.get("provider", "echo")).strip().lower()

    if provider == "echo":
        return EchoLLMService()

    if provider == "kimi":
        kimi_cfg = llm_cfg.get("kimi", {})

        api_key_env = str(kimi_cfg.get("api_key_env", "MOONSHOT_API_KEY")).strip()
        api_key = os.environ.get(api_key_env, "").strip()

        # 本地临时测试可以在 yaml 里写 api_key，但不推荐
        if not api_key:
            api_key = str(kimi_cfg.get("api_key", "")).strip()

        return KimiLLMService(
            api_key=api_key,
            base_url=str(kimi_cfg.get("base_url", "https://api.moonshot.cn/v1")),
            model=str(kimi_cfg.get("model", "kimi-k2.6")),
            system_prompt=str(kimi_cfg.get(
                "system_prompt",
                "你是一个实时数字人助手，请用自然、简洁的中文回答用户。回答适合直接转成语音播报，不要使用 Markdown。",
            )),
            max_tokens=int(kimi_cfg.get("max_tokens", 512)),
            max_history_turns=int(kimi_cfg.get("max_history_turns", 5)),
            timeout=int(kimi_cfg.get("timeout", 60)),
            disable_proxy=bool(kimi_cfg.get("disable_proxy", True)),
        )

    raise ValueError(f"不支持的 LLM provider: {provider}")