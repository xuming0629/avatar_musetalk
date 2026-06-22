from pathlib import Path
from .base import BaseAvatarEngine, AvatarRequest, AvatarResult
from nets.common.media import make_mock_video


class MockAvatarEngine(BaseAvatarEngine):
    def generate(self, request: AvatarRequest) -> AvatarResult:
        make_mock_video(request.output_path, text=request.text or "XumingAvatar Mock", fps=request.fps)
        return AvatarResult(video_path=str(request.output_path), engine="mock", message="mock video generated")
