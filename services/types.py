#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class RealtimeAvatarConfig:
    device: str = "auto"
    dtype: str = "auto"

    fps: int = 25
    bbox_shift: int = 0
    extra_margin: int = 10
    parsing_mode: str = "jaw"

    left_cheek_width: int = 90
    right_cheek_width: int = 90

    result_dir: str = "./results/realtime"
    remove_last_chunk: bool = True

    config_path: str = "configs/realtime_services.yaml"


@dataclass
class StreamFrame:
    index: int
    frame: np.ndarray
    pts: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)