# MuseTalk 重构迁移计划

## 你的目标

不是直接维护官方 MuseTalk，而是把 MuseTalk 的能力拆进自己的数字人工程：

```text
音频 -> Whisper 特征 -> Avatar 预处理 -> VAE latent -> UNet -> VAE decode -> 贴回视频 -> 输出
```

## 模块对应关系

| 你的模块 | 未来要迁移的 MuseTalk 能力 |
|---|---|
| nets/audio/whisper_extractor.py | Whisper/audio_processor |
| nets/face/detector.py | face_detection、bbox、crop |
| nets/vae/sd_vae.py | VAE encode/decode |
| nets/avatar/musetalk_engine.py | 推理主流程 |
| nets/face/blender.py | paste back、mask blend |
| tools/infer_avatar.py | 官方 inference.py |
| app.py | 平台 API |

## 推荐迁移顺序

1. 先让官方 MuseTalk 1.5 单独跑通。
2. 把官方模型加载代码迁移到 `MuseTalkEngine.load_models()`。
3. 把音频特征提取迁移到 `WhisperAudioFeatureExtractor.extract()`。
4. 把人脸检测和 bbox 逻辑迁移到 `FaceDetector`。
5. 把 VAE encode/decode 迁移到 `SDVAE`。
6. 把 UNet 推理迁移到 `MuseTalkEngine.infer_unet()`。
7. 把贴回视频和 ffmpeg 合成音频迁移到 `FaceBlender` 或 `postprocess()`。
8. 最后再做 Web 平台、多任务队列、GPU 调度。

## 不建议一开始做的事

- 不要一上来训练 MuseTalk。
- 不要直接把官方所有代码混进 app.py。
- 不要把模型权重提交到 git。
- 不要把 Web、TTS、LLM、Avatar 全塞一个文件。
