from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseAudioFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, audio_path: str | Path) -> Any:
        """Extract audio features for avatar generation."""
        raise NotImplementedError
