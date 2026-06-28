#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio Demo：LLM + TTS + 实时数字人。

功能：
    输入一段文本
        ↓
    LLM 生成回复文本
        ↓
    TTS 合成语音
        ↓
    MuseTalk 根据 TTS 音频生成数字人实时画面

运行：
    PYTHONPATH=. python tools/gradio_llm_tts_avatar.py

GPU 运行：
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python tools/gradio_llm_tts_avatar.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


import argparse
import os
import sys
from pathlib import Path

import gradio as gr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from services.types import RealtimeAvatarConfig
from services.pipeline import RealtimeDigitalHumanPipeline


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


def get_pipeline(cfg: RealtimeAvatarConfig) -> RealtimeDigitalHumanPipeline:
    global PIPELINE

    if PIPELINE is None:
        PIPELINE = RealtimeDigitalHumanPipeline(cfg=cfg)

    return PIPELINE


def llm_tts_avatar_stream(
    avatar_path,
    user_text,
    device,
    fps,
    bbox_shift,
    extra_margin,
    parsing_mode,
    left_cheek_width,
    right_cheek_width,
):
    """
    Gradio 回调：
        文本 -> LLM -> TTS -> 数字人实时画面
    """

    if avatar_path is None:
        raise gr.Error("请上传人物图片或视频")

    if not user_text or not str(user_text).strip():
        raise gr.Error("请输入文本")

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

    pipe = get_pipeline(cfg)

    # 更新脸部融合参数
    pipe.avatar_service.set_face_parsing_params(
        int(left_cheek_width),
        int(right_cheek_width),
    )

    print("========== LLM + TTS + Avatar ==========")
    print("avatar:", avatar_path)
    print("user_text:", user_text)

    # 1. LLM + TTS
    assistant_text, tts_audio_path = pipe.llm_tts_audio(user_text)

    print("========== LLM Answer ==========")
    print(assistant_text)

    print("========== TTS Audio ==========")
    print(tts_audio_path)

    info = (
        f"用户输入：{user_text}\n\n"
        f"数字人回复：{assistant_text}\n\n"
        f"TTS音频：{tts_audio_path}"
    )

    # 2. TTS 音频驱动数字人
    yielded = False

    for item in pipe.stream_audio_file(
        avatar_path=avatar_path,
        audio_path=tts_audio_path,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yielded = True
        yield info, tts_audio_path, item.frame

    if not yielded:
        yield info, tts_audio_path, None


def direct_tts_avatar_stream(
    avatar_path,
    text,
    device,
    fps,
    bbox_shift,
    extra_margin,
    parsing_mode,
    left_cheek_width,
    right_cheek_width,
):
    """
    Gradio 回调：
        文本 -> TTS -> 数字人实时画面
    不经过 LLM。
    """

    if avatar_path is None:
        raise gr.Error("请上传人物图片或视频")

    if not text or not str(text).strip():
        raise gr.Error("请输入要播报的文本")

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

    pipe = get_pipeline(cfg)

    pipe.avatar_service.set_face_parsing_params(
        int(left_cheek_width),
        int(right_cheek_width),
    )

    print("========== Direct TTS + Avatar ==========")
    print("avatar:", avatar_path)
    print("text:", text)

    # 直接 TTS
    tts_audio_path = pipe._tts_synthesize(text=text)

    info = (
        f"播报文本：{text}\n\n"
        f"TTS音频：{tts_audio_path}"
    )

    yielded = False

    for item in pipe.stream_audio_file(
        avatar_path=avatar_path,
        audio_path=tts_audio_path,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yielded = True
        yield info, tts_audio_path, item.frame

    if not yielded:
        yield info, tts_audio_path, None


def clear_history():
    global PIPELINE

    if PIPELINE is not None:
        PIPELINE.clear_history()

    return "对话历史已清空"


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# LLM + TTS + 实时数字人")
        gr.Markdown(
            "当前流程：输入文本 -> LLM 生成回复 -> TTS 合成语音 -> MuseTalk 实时生成数字人画面。"
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

        with gr.Tab("1. 文本问答数字人 LLM + TTS"):
            with gr.Row():
                with gr.Column():
                    avatar1 = gr.File(
                        label="人物图片或视频",
                        type="filepath",
                    )

                    text1 = gr.Textbox(
                        label="用户输入文本",
                        value="你好，我叫李雷，1+1等于多少？",
                        lines=4,
                    )

                    btn1 = gr.Button("开始生成数字人")
                    clear_btn = gr.Button("清空对话历史")

                with gr.Column():
                    info1 = gr.Textbox(
                        label="LLM / TTS 中间结果",
                        lines=8,
                    )

                    audio_out1 = gr.Audio(
                        label="TTS 合成音频",
                        type="filepath",
                    )

                    image_out1 = gr.Image(
                        label="实时数字人画面",
                        type="numpy",
                    )

            btn1.click(
                fn=llm_tts_avatar_stream,
                inputs=[
                    avatar1,
                    text1,
                    device,
                    fps,
                    bbox_shift,
                    extra_margin,
                    parsing_mode,
                    left_cheek_width,
                    right_cheek_width,
                ],
                outputs=[
                    info1,
                    audio_out1,
                    image_out1,
                ],
            )

            clear_btn.click(
                fn=clear_history,
                inputs=[],
                outputs=info1,
            )

        with gr.Tab("2. 直接文本播报 TTS + 数字人"):
            with gr.Row():
                with gr.Column():
                    avatar2 = gr.File(
                        label="人物图片或视频",
                        type="filepath",
                    )

                    text2 = gr.Textbox(
                        label="直接播报文本，不经过 LLM",
                        value="你好，我是实时数字人，现在正在测试文本转语音和数字人驱动。",
                        lines=4,
                    )

                    btn2 = gr.Button("开始 TTS 播报数字人")

                with gr.Column():
                    info2 = gr.Textbox(
                        label="TTS 中间结果",
                        lines=6,
                    )

                    audio_out2 = gr.Audio(
                        label="TTS 合成音频",
                        type="filepath",
                    )

                    image_out2 = gr.Image(
                        label="实时数字人画面",
                        type="numpy",
                    )

            btn2.click(
                fn=direct_tts_avatar_stream,
                inputs=[
                    avatar2,
                    text2,
                    device,
                    fps,
                    bbox_shift,
                    extra_margin,
                    parsing_mode,
                    left_cheek_width,
                    right_cheek_width,
                ],
                outputs=[
                    info2,
                    audio_out2,
                    image_out2,
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