from pathlib import Path
import numpy as np
from .base import BaseAudioFeatureExtractor


class WhisperAudioFeatureExtractor(BaseAudioFeatureExtractor):
    """
    Whisper 特征提取占位实现。
    后续把 MuseTalk 官方 whisper/audio_processor 相关代码迁移到这里。
    """

    def __init__(self, model_root: str | Path, device: str = "cuda"):
        self.model_root = Path(model_root)
        self.device = device
        self.model = None

    def load(self):
        # TODO: 加载 Whisper 模型
        self.model = "mock_whisper"
        return self

    def extract(self, audio_path: str | Path):
        if self.model is None:
            self.load()
        # TODO: 返回 MuseTalk 需要的 whisper_chunks
        return np.zeros((10, 50, 384), dtype=np.float32)
