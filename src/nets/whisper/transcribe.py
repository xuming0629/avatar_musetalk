#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : transcribe.py
# @Time          : 2026-06-24 09:35:05
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : TODO
# @Company       : 2026 XuMing. All Rights Reserved.
"""


from typing import Optional, Union

import numpy as np
import torch
import tqdm

from src.nets.whisper import N_FRAMES
from src.nets.whisper.audio import pad_or_trim, log_mel_spectrogram


"""
Whisper encoder feature extraction.

说明：
    该 transcribe 不做文本解码，只提取 encoder embeddings。
    输出结构保持和原代码一致：
        {
            "segments": [
                {
                    "start": ...,
                    "end": ...,
                    "encoder_embeddings": ...
                }
            ]
        }
"""

from typing import Optional, Union

import numpy as np
import torch
import tqdm

from src.nets.whisper import N_FRAMES
from src.nets.whisper.audio import pad_or_trim, log_mel_spectrogram


def transcribe(
    model,
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    fp16: bool = True,
):
    """
    提取 Whisper encoder embeddings。

    Args:
        model: Whisper 模型
        audio: 音频路径、np.ndarray 或 torch.Tensor
        verbose: 是否显示进度条
        fp16: CUDA 下是否使用 float16

    Returns:
        dict:
            {
                "segments": [
                    {
                        "start": int,
                        "end": int,
                        "encoder_embeddings": np.ndarray
                    }
                ]
            }
    """
    dtype = torch.float16 if fp16 else torch.float32

    if model.device == torch.device("cpu"):
        dtype = torch.float32

    mel = log_mel_spectrogram(
        audio
    )

    num_frames = mel.shape[-1]
    seek = 0
    segments = []

    with tqdm.tqdm(
        total=num_frames,
        unit="frames",
        disable=verbose is not False,
    ) as pbar:

        while seek < num_frames:
            end_seek = min(
                seek + N_FRAMES,
                num_frames
            )

            segment = pad_or_trim(
                mel[:, seek:end_seek],
                N_FRAMES
            )

            segment = segment.to(
                device=model.device,
                dtype=dtype
            )

            if segment.ndim == 2:
                segment = segment.unsqueeze(0)

            with torch.no_grad():
                _, embeddings = model.encoder(
                    segment,
                    include_embeddings=True
                )

            segments.append(
                {
                    "start": seek,
                    "end": end_seek,
                    "encoder_embeddings": embeddings,
                }
            )

            pbar.update(
                end_seek - seek
            )

            seek = end_seek

    return {
        "segments": segments
    }