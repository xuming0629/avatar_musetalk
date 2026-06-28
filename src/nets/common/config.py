#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import torch


def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path)

    print("[Config] load:", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def resolve_path(
    path: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None,
) -> str:
    path = Path(path)

    if path.is_absolute():
        return str(path)

    if project_root is not None:
        return str(Path(project_root) / path)

    return str(path)


def normalize_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    统一处理 device：
    - None -> 自动 cuda/cpu
    - "cuda" / "cuda:0" -> CUDA 可用才使用，否则回退 CPU
    - "cpu" -> CPU
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_str = str(device).strip().lower()

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available, fallback to CPU")
            return torch.device("cpu")

        return torch.device(device_str)

    if device_str == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device}. Expected cpu/cuda/cuda:0")


def get_device(config_path: str = "configs/musetalk_v15.yaml") -> torch.device:
    cfg = load_yaml(config_path)

    device = cfg.get("runtime", {}).get("device", None)

    return normalize_device(device)