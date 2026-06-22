from pathlib import Path
from nets.avatar.base import AvatarRequest, AvatarResult
from nets.avatar.mock_engine import MockAvatarEngine
from nets.avatar.musetalk_engine import MuseTalkEngine
from nets.common.media import ensure_dir


class DigitalHumanPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_dir = ensure_dir(cfg.get("paths", {}).get("output_dir", "assets/outputs"))
        use_mock = cfg.get("runtime", {}).get("use_mock", True)
        self.engine = MockAvatarEngine() if use_mock else MuseTalkEngine(cfg)

    def generate(self, avatar_path: str, audio_path: str, text: str | None = None) -> AvatarResult:
        out_path = self.output_dir / "result.mp4"
        req = AvatarRequest(
            avatar_path=avatar_path,
            audio_path=audio_path,
            output_path=str(out_path),
            fps=self.cfg.get("runtime", {}).get("fps", 25),
            text=text,
        )
        return self.engine.generate(req)
