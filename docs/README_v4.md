# avatar v4.0 流水线实时回答补丁

## 目标

在不修改 MuseTalk / VAE / UNet / Whisper / FaceParsing 底层模型的前提下，
把 v3 的“分句串行流式”升级为 v4 的“三段流水线流式”。

## v3 的问题

v3 是：

```text
LLM 分句
-> 当前句 TTS
-> 当前句 Whisper + MuseTalk
-> 播放当前句
-> 再处理下一句
```

所以句子之间会卡顿。

## v4 的结构

v4 改成：

```text
LLM worker:
    token 流式输出 -> 合并短句 -> text_queue

TTS worker:
    text_queue -> 提前合成 wav -> audio_queue

Avatar worker:
    audio_queue -> 预生成几帧 -> 追加音频 -> MuseTalk 按帧推送
```

这样数字人播放当前段时，TTS 可以提前准备下一段，减少段间等待。

## 关键参数

在 `tools/webrtc_avatar_server_stream_v4.py` 顶部：

```python
MAX_STREAM_SENTENCE_CHARS = 80
MIN_STREAM_SENTENCE_CHARS = 18
TEXT_QUEUE_MAX = 32
TTS_QUEUE_MAX = 4
AVATAR_PREBUFFER_FRAMES = 6
```

如果想更快首包，可以减小 `MIN_STREAM_SENTENCE_CHARS`；
如果想更流畅，可以增大 `MIN_STREAM_SENTENCE_CHARS` 或 `AVATAR_PREBUFFER_FRAMES`。

## 安装

在项目根目录执行：

```bash
unzip -o avatar_v4_streaming_pipeline_patch.zip -d /home/xuming/avatar_musetalk
```

## 启动

建议先用 15 FPS 测试流畅度：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python -u tools/webrtc_avatar_server_stream_v4.py \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --fps 15
```

如果 15 FPS 流畅，再尝试：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python -u tools/webrtc_avatar_server_stream_v4.py \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --fps 20
```

3060 12G 如果 MuseTalk 单帧速度跟不上，25 FPS 可能仍然卡顿。
