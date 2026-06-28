#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio Demo：先加载数字人，再输入问题让数字人回答。

运行：
    PYTHONPATH=. python tools/gradio_loaded_avatar_demo.py

GPU：
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python tools/gradio_loaded_avatar_demo.py
"""

from __future__ import annotations

import argparse
import os
import sys

import gradio as gr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from services.pipeline_v2 import RealtimeDigitalHumanPipeline
from services.types import RealtimeAvatarConfig


PIPELINE: RealtimeDigitalHumanPipeline | None = None


def build_cfg(
    device: str,
    fps: int,
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
) -> RealtimeAvatarConfig:
    return RealtimeAvatarConfig(
        device=device,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        left_cheek_width=int(left_cheek_width),
        right_cheek_width=int(right_cheek_width),
        config_path="configs/realtime_services.yaml",
    )


def load_avatar_ui(
    avatar_path,
    device,
    fps,
    bbox_shift,
    extra_margin,
    parsing_mode,
    left_cheek_width,
    right_cheek_width,
):
    """加载数字人，并先展示 idle frame。"""

    global PIPELINE

    if avatar_path is None:
        raise gr.Error("请先上传人物图片或视频")

    if not os.path.exists(avatar_path):
        raise gr.Error(f"人物图片/视频不存在: {avatar_path}")

    cfg = build_cfg(
        device=device,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        left_cheek_width=int(left_cheek_width),
        right_cheek_width=int(right_cheek_width),
    )

    PIPELINE = RealtimeDigitalHumanPipeline(cfg=cfg)

    PIPELINE.avatar_service.set_face_parsing_params(
        int(left_cheek_width),
        int(right_cheek_width),
    )

    idle_frame = PIPELINE.load_avatar(
        avatar_path=avatar_path,
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
    )

    info = (
        "数字人已加载完成。\n"
        "现在可以在下面输入问题，数字人会直接开口回答。\n"
        f"avatar: {avatar_path}"
    )

    return info, idle_frame


def ask_avatar_ui(
    user_text,
    reference_audio,
    fps,
    parsing_mode,
):
    """用户输入问题，已加载数字人回答。"""

    global PIPELINE

    if PIPELINE is None:
        raise gr.Error("请先点击“加载数字人”")

    if not user_text or not str(user_text).strip():
        raise gr.Error("请输入问题")

    yielded = False
    last_frame = None
    info = ""

    for assistant_text, tts_audio_path, frame in PIPELINE.answer_with_loaded_avatar(
        user_text=user_text,
        reference_audio=reference_audio,
        fps=int(fps),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yielded = True
        last_frame = frame

        info = (
            f"用户：{user_text}\n\n"
            f"数字人：{assistant_text}\n\n"
            f"TTS音频：{tts_audio_path}"
        )

        yield info, tts_audio_path, frame

    if not yielded:
        idle_frame = PIPELINE.get_idle_frame()
        yield "没有生成数字人画面", None, idle_frame

    else:
        # 回答结束后停留在最后一帧
        yield info, tts_audio_path, last_frame


def tts_avatar_ui(
    text,
    reference_audio,
    fps,
    parsing_mode,
):
    """不经过 LLM，直接文本 TTS 播报。"""

    global PIPELINE

    if PIPELINE is None:
        raise gr.Error("请先点击“加载数字人”")

    if not text or not str(text).strip():
        raise gr.Error("请输入播报文本")

    yielded = False
    last_frame = None
    info = ""

    for spoken_text, tts_audio_path, frame in PIPELINE.stream_text_with_loaded_avatar(
        text=text,
        reference_audio=reference_audio,
        fps=int(fps),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yielded = True
        last_frame = frame

        info = (
            f"播报文本：{spoken_text}\n\n"
            f"TTS音频：{tts_audio_path}"
        )

        yield info, tts_audio_path, frame

    if not yielded:
        idle_frame = PIPELINE.get_idle_frame()
        yield "没有生成数字人画面", None, idle_frame

    else:
        yield info, tts_audio_path, last_frame


def clear_history_ui():
    global PIPELINE

    if PIPELINE is not None:
        PIPELINE.clear_history()
        return "对话历史已清空"

    return "数字人还没有加载"


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# XumingAvatar：先展示数字人，再输入问题让它回答")
        gr.Markdown(
            "流程：上传人物 -> 加载数字人并显示 -> 输入问题 -> LLM -> TTS -> 数字人回答。"
        )

        with gr.Accordion("通用参数", open=True):
            with gr.Row():
                device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="cpu",
                    label="device",
                )

                fps = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=25,
                    step=1,
                    label="fps",
                )

                parsing_mode = gr.Dropdown(
                    choices=["jaw", "raw", "face"],
                    value="jaw",
                    label="parsing_mode",
                )

            with gr.Row():
                bbox_shift = gr.Slider(
                    minimum=-20,
                    maximum=20,
                    value=0,
                    step=1,
                    label="bbox_shift",
                )

                extra_margin = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=10,
                    step=1,
                    label="extra_margin",
                )

                left_cheek_width = gr.Slider(
                    minimum=0,
                    maximum=150,
                    value=90,
                    step=1,
                    label="left_cheek_width",
                )

                right_cheek_width = gr.Slider(
                    minimum=0,
                    maximum=150,
                    value=90,
                    step=1,
                    label="right_cheek_width",
                )

        with gr.Row():
            with gr.Column():
                avatar = gr.File(
                    label="人物图片或视频",
                    type="filepath",
                )

                load_btn = gr.Button("1. 加载数字人")

                status = gr.Textbox(
                    label="状态 / 中间结果",
                    lines=8,
                )

            with gr.Column():
                avatar_view = gr.Image(
                    label="数字人画面",
                    type="numpy",
                )

        load_btn.click(
            fn=load_avatar_ui,
            inputs=[
                avatar,
                device,
                fps,
                bbox_shift,
                extra_margin,
                parsing_mode,
                left_cheek_width,
                right_cheek_width,
            ],
            outputs=[
                status,
                avatar_view,
            ],
        )

        with gr.Tab("文本问答：LLM + TTS + 数字人回答"):
            with gr.Row():
                with gr.Column():
                    question = gr.Textbox(
                        label="输入问题",
                        value="你好，我叫李雷，1+1等于多少？",
                        lines=4,
                    )

                    ref_audio = gr.Audio(
                        label="TTS 参考音频，可选",
                        type="filepath",
                    )

                    ask_btn = gr.Button("2. 发送问题，让数字人回答")
                    clear_btn = gr.Button("清空对话历史")

                with gr.Column():
                    tts_audio = gr.Audio(
                        label="TTS 合成音频",
                        type="filepath",
                    )

            ask_btn.click(
                fn=ask_avatar_ui,
                inputs=[
                    question,
                    ref_audio,
                    fps,
                    parsing_mode,
                ],
                outputs=[
                    status,
                    tts_audio,
                    avatar_view,
                ],
            )

            clear_btn.click(
                fn=clear_history_ui,
                inputs=[],
                outputs=status,
            )

        with gr.Tab("直接播报：TTS + 数字人"):
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(
                        label="直接播报文本",
                        value="你好，我是实时数字人，现在已经加载完成，可以开始播报。",
                        lines=4,
                    )

                    ref_audio2 = gr.Audio(
                        label="TTS 参考音频，可选",
                        type="filepath",
                    )

                    tts_btn = gr.Button("直接 TTS 播报")

                with gr.Column():
                    tts_audio2 = gr.Audio(
                        label="TTS 合成音频",
                        type="filepath",
                    )

            tts_btn.click(
                fn=tts_avatar_ui,
                inputs=[
                    tts_text,
                    ref_audio2,
                    fps,
                    parsing_mode,
                ],
                outputs=[
                    status,
                    tts_audio2,
                    avatar_view,
                ],
            )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()