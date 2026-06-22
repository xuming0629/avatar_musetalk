#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nets.common.config import load_config
from nets.avatar.base import AvatarRequest
from nets.avatar.musetalk_engine import MuseTalkEngine


def parse_args():
    parser = argparse.ArgumentParser("XumingAvatar inference")
    parser.add_argument("--config", default="configs/musetalk_v15.yaml")
    parser.add_argument("--avatar", required=True, help="avatar image/video path")
    parser.add_argument("--audio", required=True, help="audio path")
    parser.add_argument("--out", default="assets/outputs/result.mp4")
    parser.add_argument("--mock", action="store_true", help="force mock output")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.mock:
        cfg.setdefault("runtime", {})["use_mock"] = True
    engine = MuseTalkEngine(cfg)
    req = AvatarRequest(
        avatar_path=args.avatar,
        audio_path=args.audio,
        output_path=args.out,
        fps=cfg.get("runtime", {}).get("fps", 25),
        text="XumingAvatar",
    )
    result = engine.generate(req)
    print(result)


if __name__ == "__main__":
    main()
