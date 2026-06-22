# XumingAvatar / 灵犀数字人引擎

一个面向二次开发的模块化 2D 数字人项目骨架，目标是把 MuseTalk 拆成可插拔模块，后续可替换为 EchoMimic、LivePortrait、Audio2Face 等引擎。

## 目录

```text
xuming_avatar/
├── assets/                 # 输入输出资源
│   ├── avatars/            # 数字人图片/视频
│   ├── audios/             # 音频 wav/mp3
│   ├── outputs/            # 生成结果
│   └── temp/               # 临时文件
├── configs/                # 配置文件
├── models/                 # 模型权重目录，不放入 git
│   ├── musetalk-v1.5/
│   ├── whisper/
│   ├── sd-vae-ft-mse/
│   ├── face-parse-bisent/
│   ├── resnet18/
│   └── dwpose/
├── nets/                   # 核心算法模块
│   ├── audio/              # Whisper 音频特征
│   ├── face/               # 人脸检测、裁剪、融合
│   ├── vae/                # VAE 编码解码
│   ├── avatar/             # 数字人引擎
│   └── common/             # 通用工具
├── services/               # 平台服务层
├── tools/                  # 命令行工具
├── app.py                  # FastAPI 入口
└── requirements.txt
```

## 快速启动

```bash
conda create -n xavatar python=3.10 -y
conda activate xavatar
pip install -r requirements.txt
python app.py
```

访问：

```text
http://127.0.0.1:8000/docs
```

## 离线推理

```bash
python tools/infer_avatar.py \
  --config configs/musetalk_v15.yaml \
  --avatar assets/avatars/demo.mp4 \
  --audio assets/audios/demo.wav \
  --out assets/outputs/demo.mp4
```

当前版本默认使用 Mock 引擎生成占位视频。你后续只需要实现：

```text
nets/avatar/musetalk_engine.py
```

里面的真实 MuseTalk 推理逻辑即可。
