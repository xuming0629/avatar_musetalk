#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_tts.py
# @Time          : 2026-06-25 17:19:02
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TTS 语言合成功能测试
# @Company       : 2026 XuMing. All Rights Reserved.
"""




import os
import sys
import cv2 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 TTS factory 调用。

运行：
    PYTHONPATH=. python tests/test_tts.py
"""

import os
import sys
from pathlib import Path

import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from services.tts import build_tts_service


def load_yaml_config(config_path: str):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


def main():
    cfg = load_yaml_config("configs/realtime_services.yaml")

    tts = build_tts_service(cfg)

    print("========== TTS Factory Test ==========")
    print("tts service:", tts.__class__.__name__)
    print("tts name:", getattr(tts, "name", "unknown"))

    text = "你好，我是实时数字人，现在正在测试通过配置文件调用语音合成功能。"

    out = tts.synthesize(text)

    print("tts output:", out)

    if not os.path.exists(out):
        raise RuntimeError(f"TTS 输出文件不存在: {out}")

    print("TTS 测试成功")


if __name__ == "__main__":
    main()