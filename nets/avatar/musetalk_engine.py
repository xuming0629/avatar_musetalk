from pathlib import Path
from .base import BaseAvatarEngine, AvatarRequest, AvatarResult
from nets.audio.whisper_extractor import WhisperAudioFeatureExtractor
from nets.face.detector import FaceDetector
from nets.face.blender import FaceBlender
from nets.vae.sd_vae import SDVAE
from nets.common.media import read_video_frames, read_image_as_frame, write_video, make_mock_video


class MuseTalkEngine(BaseAvatarEngine):
    """
    你自己的 MuseTalk 模块化实现入口。

    目标拆分：
    1. load_models()        加载 Whisper / VAE / UNet / FaceParser
    2. preprocess_avatar()  人脸检测、裁剪、mask、latent 缓存
    3. extract_audio()      Whisper 音频特征
    4. infer_unet()         UNet 生成新嘴型 latent
    5. decode_faces()       VAE decode 成图像
    6. postprocess()        贴回原视频、合成音频

    当前为了让工程可运行，默认 fallback 到 mock 视频。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        runtime = cfg.get("runtime", {})
        models = cfg.get("models", {})
        self.device = runtime.get("device", "cuda")
        self.use_mock = runtime.get("use_mock", True)
        self.audio_extractor = WhisperAudioFeatureExtractor(models.get("whisper", {}).get("root", "models/whisper"), self.device)
        self.face_detector = FaceDetector(self.device)
        self.face_blender = FaceBlender()
        self.vae = SDVAE(models.get("vae", {}).get("root", "models/sd-vae-ft-mse"), self.device)
        self.unet = None

    def load_models(self):
        # TODO: 加载 MuseTalk 1.5 UNet 权重
        self.audio_extractor.load()
        self.vae.load()
        self.unet = "mock_musetalk_unet"
        return self

    def preprocess_avatar(self, avatar_path: str | Path):
        path = Path(avatar_path)
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            frame = read_image_as_frame(path)
            frames = [frame]
            fps = self.cfg.get("runtime", {}).get("fps", 25)
        else:
            frames, fps = read_video_frames(path)
        if not frames:
            raise RuntimeError(f"empty avatar input: {path}")
        bboxes = [self.face_detector.detect(f) for f in frames]
        return frames, bboxes, fps

    def infer_unet(self, face_latents, audio_features):
        # TODO: 使用 MuseTalk UNet 真实推理
        return face_latents

    def postprocess(self, frames, bboxes, generated_faces, out_path: str | Path, fps: int):
        out_frames = []
        for i, frame in enumerate(frames):
            face = generated_faces[i % len(generated_faces)]
            bbox = bboxes[i % len(bboxes)]
            out_frames.append(self.face_blender.paste_back(frame, face, bbox))
        return write_video(out_frames, out_path, fps=fps)

    def generate(self, request: AvatarRequest) -> AvatarResult:
        if self.unet is None:
            self.load_models()

        if self.use_mock:
            make_mock_video(request.output_path, text=request.text or "MuseTalk Mock", fps=request.fps)
            return AvatarResult(video_path=str(request.output_path), engine="musetalk-mock")

        frames, bboxes, input_fps = self.preprocess_avatar(request.avatar_path)
        audio_features = self.audio_extractor.extract(request.audio_path)
        face_crops = [self.face_detector.crop_face(f, b) for f, b in zip(frames, bboxes)]
        latents = self.vae.encode(face_crops)
        pred_latents = self.infer_unet(latents, audio_features)
        generated_faces = self.vae.decode(pred_latents)
        video_path = self.postprocess(frames, bboxes, generated_faces, request.output_path, request.fps or input_fps)
        return AvatarResult(video_path=str(video_path), engine="musetalk-v1.5")
