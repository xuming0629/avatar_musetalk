#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : base.py
# @Time          : 2026-06-22 18:18:12
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TODO
# @Company       : 2026 XuMing. All Rights Reserved.
"""




from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseAudioFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, audio_path: str | Path) -> Any:
        """Extract audio features for avatar generation."""
        raise NotImplementedError
