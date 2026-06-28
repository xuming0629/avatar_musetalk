#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试：LLM + TTS + 实时数字人。

运行：
    PYTHONPATH=. python tests/test_llm_tts_avatar.py \
      --avatar ./assets/3456.png \
      --text "你好，我叫李雷，1+1等于多少？"
"""

import os
import sys
import argparse
from pathlib import Path

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from services.types import RealtimeAvatarConfig
from services.pipeline import RealtimeDigitalHumanPipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--avatar",
        type=str,
        default="./assets/3456.png",
        help="人物图片或视频路径",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="你好，我叫李雷，1+1等于多少？",
        help="用户输入文本，会先送给 LLM",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/output/llm_tts_avatar_frames",
        help="保存输出帧目录",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--parsing_mode", type=str, default="jaw")

    args = parser.parse_args()

    if not os.path.exists(args.avatar):
        raise FileNotFoundError(f"人物图片/视频不存在: {args.avatar}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RealtimeAvatarConfig(
        device=args.device,
        fps=args.fps,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        config_path="configs/realtime_services.yaml",
    )

    pipe = RealtimeDigitalHumanPipeline(cfg=cfg)

    print("========== LLM + TTS + Avatar Test ==========")
    print("avatar:", args.avatar)
    print("user text:", args.text)
    print("out_dir:", str(out_dir))

    # 先生成 LLM 回复和 TTS 音频，方便你看到中间结果
    assistant_text, tts_audio_path = pipe.llm_tts_audio(args.text)

    print("========== LLM Answer ==========")
    print(assistant_text)

    print("========== TTS Audio ==========")
    print(tts_audio_path)

    frame_count = 0

    for item in pipe.stream_audio_file(
        avatar_path=args.avatar,
        audio_path=tts_audio_path,
        fps=args.fps,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        realtime_sleep=False,
    ):
        frame = item.frame
        save_path = out_dir / f"{frame_count:08d}.png"

        # frame 一般是 RGB，cv2.imwrite 需要 BGR
        cv2.imwrite(
            str(save_path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )

        frame_count += 1

    print("========== Result ==========")
    print("frame_count:", frame_count)
    print("frames:", out_dir)

    if frame_count == 0:
        raise RuntimeError("没有生成任何数字人帧")

    print("测试成功")


if __name__ == "__main__":
    main()