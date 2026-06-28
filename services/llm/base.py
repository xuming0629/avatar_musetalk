#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

from openai import OpenAI


Message = Dict[str, str]


class BaseLLMService:
    """LLM 服务接口。"""

    name = "base"

    def chat(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> str:
        raise NotImplementedError

    def stream_chat(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> Iterator[str]:
        """流式返回 LLM 文本。

        默认实现：兼容旧的非流式 LLM，直接一次性 yield chat() 结果。
        具体服务可以覆盖这个方法，实现真正 token 级流式返回。
        """
        answer = self.chat(
            user_text=user_text,
            history=history,
        )

        if answer:
            yield answer


class EchoLLMService(BaseLLMService):
    """占位 LLM，用来跑通流程。"""

    name = "echo"

    def chat(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return "我没有听清楚，请你再说一遍。"
        return f"我听到了：{user_text}"

    def stream_chat(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> Iterator[str]:
        answer = self.chat(
            user_text=user_text,
            history=history,
        )

        # 模拟轻量流式，便于本地测试整条链路。
        for ch in answer:
            yield ch


class KimiLLMService(BaseLLMService):
    """Kimi / Moonshot LLM 服务。

    这个类只做一件事：
    输入用户文本，调用 Kimi API，返回回复文本。
    """

    name = "kimi"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-k2.6",
        system_prompt: str = "你是一个实时数字人助手，请用自然、简洁的中文回答用户。回答适合直接转成语音播报，不要使用 Markdown。",
        max_tokens: int = 512,
        max_history_turns: int = 5,
        timeout: int = 60,
        disable_proxy: bool = True,
    ):
        if disable_proxy:
            self._clear_proxy_env()

        self.api_key = (api_key or os.environ.get("MOONSHOT_API_KEY", "")).strip()
        self.base_url = (base_url or "https://api.moonshot.cn/v1").strip().rstrip("/")
        self.model = (model or "kimi-k2.6").strip()
        self.system_prompt = system_prompt or ""
        self.max_tokens = int(max_tokens)
        self.max_history_turns = int(max_history_turns)

        if not self.api_key:
            raise RuntimeError(
                "没有读取到 Kimi API Key，请先设置：\n"
                "export MOONSHOT_API_KEY='你的Kimi API Key'"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    def chat(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> str:
        user_text = (user_text or "").strip()

        if not user_text:
            return "我没有听清楚，请你再说一遍。"

        messages = self._build_messages(user_text, history)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # Kimi k2.6 当前 temperature 只能为 1
            temperature=1,
            max_completion_tokens=self.max_tokens,
        )

        answer = completion.choices[0].message.content

        if not answer:
            return "我暂时没有想到合适的回答。"

        return str(answer).strip()

    def stream_chat(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> Iterator[str]:
        """Kimi / Moonshot token 级流式返回。

        注意：
            这里只负责返回文本 token，不更新 history。
            history 由上层 pipeline/server 在最终回答完成后统一更新。
        """
        user_text = (user_text or "").strip()

        if not user_text:
            yield "我没有听清楚，请你再说一遍。"
            return

        messages = self._build_messages(user_text, history)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            max_completion_tokens=self.max_tokens,
            stream=True,
        )

        has_content = False

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if not content:
                continue

            has_content = True
            yield str(content)

        if not has_content:
            yield "我暂时没有想到合适的回答。"

    def _build_messages(
        self,
        user_text: str,
        history: Optional[List[Message]] = None,
    ) -> List[Message]:
        messages: List[Message] = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        if history:
            recent = history[-self.max_history_turns * 2:]
            for item in recent:
                role = item.get("role", "")
                content = item.get("content", "")

                if role not in ("user", "assistant"):
                    continue

                if not content:
                    continue

                messages.append({
                    "role": role,
                    "content": content,
                })

        messages.append({
            "role": "user",
            "content": user_text,
        })

        return messages

    @staticmethod
    def _clear_proxy_env():
        """清理 127.0.0.1:7890 这类坏代理。"""
        for key in [
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
        ]:
            os.environ.pop(key, None)