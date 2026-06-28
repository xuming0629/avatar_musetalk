#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 services 的实时数字人 Gradio Demo。

运行：
    PYTHONPATH=. python tools/realtime_services_demo.py

这份 demo 只调用 services，不直接改 MuseTalk 底层算法。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from services.pipeline import RealtimeDigitalHumanPipeline
from services.types import RealtimeAvatarConfig


PIPELINE: RealtimeDigitalHumanPipeline | None = None


def get_pipeline(cfg: RealtimeAvatarConfig) -> RealtimeDigitalHumanPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = RealtimeDigitalHumanPipeline(cfg=cfg)
    return PIPELINE


def build_cfg(device: str, fps: int, bbox_shift: int, extra_margin: int, parsing_mode: str, left_cheek_width: int, right_cheek_width: int):
    return RealtimeAvatarConfig(
        device=device,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        left_cheek_width=int(left_cheek_width),
        right_cheek_width=int(right_cheek_width),
    )


def stream_audio_file_ui(avatar_path, audio_path, device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width):
    cfg = build_cfg(device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width)
    pipe = get_pipeline(cfg)
    pipe.avatar_service.set_face_parsing_params(int(left_cheek_width), int(right_cheek_width))

    if avatar_path is None:
        raise gr.Error("请上传人物图片或视频")
    if audio_path is None:
        raise gr.Error("请上传音频")

    for item in pipe.stream_audio_file(
        avatar_path=avatar_path,
        audio_path=audio_path,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yield item.frame


def stream_text_ui(avatar_path, text, reference_audio, device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width):
    cfg = build_cfg(device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width)
    pipe = get_pipeline(cfg)
    pipe.avatar_service.set_face_parsing_params(int(left_cheek_width), int(right_cheek_width))

    if avatar_path is None:
        raise gr.Error("请上传人物图片或视频")
    if not text:
        raise gr.Error("请输入要播报的文本")

    for item in pipe.stream_text(
        avatar_path=avatar_path,
        text=text,
        reference_audio=reference_audio,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yield item.frame


def conversation_ui(avatar_path, user_audio, reference_audio, device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width):
    cfg = build_cfg(device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width)
    pipe = get_pipeline(cfg)
    pipe.avatar_service.set_face_parsing_params(int(left_cheek_width), int(right_cheek_width))

    if avatar_path is None:
        raise gr.Error("请上传人物图片或视频")
    if user_audio is None:
        raise gr.Error("请上传或录制用户语音")

    # 先跑 ASR/LLM/TTS，拿到文本结果；然后再逐帧显示数字人。
    conv = pipe.run_conversation_audio(audio_path=user_audio, reference_audio=reference_audio)

    # gradio 多输出：第一个输出文本，第二个输出图像流。
    # generator 需要每一轮 yield 两个输出。
    first_text = f"用户：{conv.user_text}\n数字人：{conv.assistant_text}\nTTS音频：{conv.tts_audio_path}"
    yielded = False
    for item in pipe.stream_audio_file(
        avatar_path=avatar_path,
        audio_path=conv.tts_audio_path,
        fps=int(fps),
        bbox_shift=int(bbox_shift),
        extra_margin=int(extra_margin),
        parsing_mode=parsing_mode,
        realtime_sleep=True,
    ):
        yielded = True
        yield first_text, item.frame

    if not yielded:
        yield first_text, None


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# XumingAvatar 实时数字人服务框架 Demo")
        gr.Markdown(
            "底层 MuseTalk 算法不动，ASR / LLM / TTS / 麦克风 / 推流都已经拆到 services，后续直接替换对应 service。"
        )

        with gr.Accordion("通用参数", open=True):
            with gr.Row():
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="device")
                fps = gr.Slider(5, 30, value=25, step=1, label="fps")
                parsing_mode = gr.Dropdown(["jaw", "raw", "face"], value="jaw", label="parsing_mode")
            with gr.Row():
                bbox_shift = gr.Slider(-20, 20, value=0, step=1, label="bbox_shift")
                extra_margin = gr.Slider(0, 50, value=10, step=1, label="extra_margin")
                left_cheek_width = gr.Slider(0, 150, value=90, step=1, label="left_cheek_width")
                right_cheek_width = gr.Slider(0, 150, value=90, step=1, label="right_cheek_width")

        with gr.Tab("1. 音频文件驱动"):
            with gr.Row():
                with gr.Column():
                    avatar1 = gr.File(label="人物图片/视频", type="filepath")
                    audio1 = gr.Audio(label="驱动音频", type="filepath")
                    btn1 = gr.Button("开始生成")
                with gr.Column():
                    out1 = gr.Image(label="实时画面", type="numpy")
            btn1.click(
                stream_audio_file_ui,
                inputs=[avatar1, audio1, device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
                outputs=out1,
            )

        with gr.Tab("2. 文本 TTS 播报"):
            with gr.Row():
                with gr.Column():
                    avatar2 = gr.File(label="人物图片/视频", type="filepath")
                    text2 = gr.Textbox(label="播报文本", value="你好，我是实时数字人。")
                    ref_audio2 = gr.Audio(label="参考音频/占位音频，可选", type="filepath")
                    btn2 = gr.Button("开始 TTS 播报")
                with gr.Column():
                    out2 = gr.Image(label="实时画面", type="numpy")
            btn2.click(
                stream_text_ui,
                inputs=[avatar2, text2, ref_audio2, device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
                outputs=out2,
            )

        with gr.Tab("3. 麦克风/录音 + ASR + LLM + TTS"):
            with gr.Row():
                with gr.Column():
                    avatar3 = gr.File(label="人物图片/视频", type="filepath")
                    user_audio3 = gr.Audio(label="用户语音：可上传或浏览器录制", type="filepath")
                    ref_audio3 = gr.Audio(label="TTS 参考音频/占位音频，可选", type="filepath")
                    btn3 = gr.Button("开始对话")
                with gr.Column():
                    text3 = gr.Textbox(label="ASR/LLM/TTS 中间结果", lines=5)
                    out3 = gr.Image(label="实时画面", type="numpy")
            btn3.click(
                conversation_ui,
                inputs=[avatar3, user_audio3, ref_audio3, device, fps, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
                outputs=[text3, out3],
            )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    demo = create_demo()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
