#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str = "configs/realtime_services.yaml") -> Dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return cfg


    
# export MOONSHOT_API_KEY="你的_kimi_key"

# PYTHONPATH=. python tools/test_llm_from_config.py

# export MOONSHOT_API_KEY=sk-GzvMd9dvRQmQZFBNvJ6SHpHCuBOWfcviMLruTTIRsYjJa6nf