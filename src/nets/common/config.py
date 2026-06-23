#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : config.py
# @Time          : 2026-06-22 14:04:41
# @Author        : XuMing
# @Email         : xuming09@inspur.com
# @description   :
# Copyright      : Shandong Inspur Software Co., Ltd. 灵犀有言
"""



from pathlib import Path
from typing import Optional, Any, Dict
import yaml
import torch 


def get_device(config_path="configs/musetalk_v15.yaml"):
    cfg = load_yaml(config_path)

    device = cfg.get("runtime", {}).get("device", "cpu")

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, fallback to CPU")
        device = "cpu"

    return device

def load_yaml(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    print(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def resolve_path(path: str, root_dir: Optional[str] = None) -> str:
    """
    支持相对路径和绝对路径。
    相对路径默认基于工程根目录。
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)

    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir)

    return str((root_dir / p).resolve())