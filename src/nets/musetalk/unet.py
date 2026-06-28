#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : unet.py
# @Time          : 2026-06-23 22:56:28
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : MuseTalk UNet wrapper with default config support
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import os
import sys
import json
import math
from typing import Optional

import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel

from src.nets.common.config import (
    load_yaml,
    resolve_path,
    get_device,
)


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.insert(0, ROOT_DIR)


def get_musetalk_unet_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
):
    """
    从配置文件读取 MuseTalk UNet 参数。

    支持配置格式：

    models:
      musetalk:
        path: models/musetalkV15/unet.pth
        cfg: models/musetalkV15/musetalk.json
        use_float16: false
    """

    cfg = load_yaml(config_path)

    model_cfg = (
        cfg.get("models", {})
           .get("musetalk", {})
    )

    model_path = model_cfg.get("path", None)
    unet_config = model_cfg.get("cfg", None)

    if model_path is None:
        raise ValueError(
            "Config error: models.musetalk.path not found"
        )

    if unet_config is None:
        raise ValueError(
            "Config error: models.musetalk.cfg not found"
        )

    model_path = resolve_path(
        model_path,
        project_root,
    )

    unet_config = resolve_path(
        unet_config,
        project_root,
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MuseTalk UNet weight not found: {model_path}"
        )

    if not os.path.isfile(unet_config):
        raise FileNotFoundError(
            f"MuseTalk UNet config not found: {unet_config}"
        )

    use_float16 = model_cfg.get(
        "use_float16",
        False,
    )

    return {
        "model_path": model_path,
        "unet_config": unet_config,
        "use_float16": use_float16,
    }


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        max_len: int = 5000,
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model,
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term,
        )

        pe[:, 1::2] = torch.cos(
            position * div_term,
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )


    def _run_without_cudnn(self, fn):
        """
        UNet CUDA 推理专用封装。

        说明:
            - CPU 时保持原逻辑；
            - CUDA 时仍然走 GPU，不会回退到 CPU；
            - 仅在 UNet forward 期间临时关闭 cuDNN；
            - forward 结束后恢复原来的 cuDNN 设置，避免影响其他模型。
        """
        if "cuda" not in str(self.device):
            with torch.inference_mode():
                return fn()

        old_cudnn_enabled = torch.backends.cudnn.enabled
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        old_cudnn_deterministic = torch.backends.cudnn.deterministic

        try:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.cuda.synchronize()

            with torch.inference_mode():
                result = fn()

            torch.cuda.synchronize()
            return result

        finally:
            torch.backends.cudnn.enabled = old_cudnn_enabled
            torch.backends.cudnn.benchmark = old_cudnn_benchmark
            torch.backends.cudnn.deterministic = old_cudnn_deterministic

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, _ = x.size()

        pe = self.pe[
            :,
            :seq_len,
            :,
        ]

        x = x + pe.to(
            device=x.device,
            dtype=x.dtype,
        )

        return x


class UNet:
    """
    MuseTalk UNet wrapper。

    支持默认直接声明：

        unet = UNet()

    等价于：

        unet = UNet.from_config("configs/musetalk_v15.yaml")

    也支持直接传模型路径：

        unet = UNet(
            unet_config="models/musetalkV15/musetalk.json",
            model_path="models/musetalkV15/unet.pth",
            device="cuda",
        )
    """

    def __init__(
        self,
        unet_config: Optional[str] = None,
        model_path: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
        use_float16: Optional[bool] = None,
        auto_to_device: bool = True,
        eval_mode: bool = True,
    ):
        self.config_path = config_path
        self.project_root = project_root

        if unet_config is None or model_path is None:
            unet_cfg = get_musetalk_unet_config(
                config_path=config_path,
                project_root=project_root,
            )

            if model_path is None:
                model_path = unet_cfg["model_path"]

            if unet_config is None:
                unet_config = unet_cfg["unet_config"]

            if use_float16 is None:
                use_float16 = unet_cfg["use_float16"]

        else:
            model_path = resolve_path(
                model_path,
                project_root,
            )

            unet_config = resolve_path(
                unet_config,
                project_root,
            )

            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"MuseTalk UNet weight not found: {model_path}"
                )

            if not os.path.isfile(unet_config):
                raise FileNotFoundError(
                    f"MuseTalk UNet config not found: {unet_config}"
                )

            if use_float16 is None:
                use_float16 = False

        if device is None:
            device = get_device(
                config_path=config_path,
            )

        self.unet_config = unet_config
        self.model_path = model_path
        self.device = torch.device(device)
        self.use_float16 = bool(use_float16)

        print(f"[UNet] load config from: {self.unet_config}")

        with open(
            self.unet_config,
            "r",
            encoding="utf-8",
        ) as f:
            unet_cfg_json = json.load(f)

        self.model = UNet2DConditionModel(
            **unet_cfg_json,
        )

        self.pe = PositionalEncoding(
            d_model=384,
        )

        print(f"[UNet] load weight from: {self.model_path}")

        try:
            weights = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            weights = torch.load(
                self.model_path,
                map_location="cpu",
            )

        self.model.load_state_dict(
            weights,
            strict=True,
        )

        if auto_to_device:
            self.model.to(
                self.device,
            )
            self.pe.to(
                self.device,
            )

        if self.use_float16:
            self.model = self.model.half()
            self.pe = self.pe.half()

        if eval_mode:
            self.model.eval()
            self.pe.eval()

        print(
            f"[UNet] load success: {self.model_path}, "
            f"device={self.device}, "
            f"use_float16={self.use_float16}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device: Optional[str] = None,
        use_float16: Optional[bool] = None,
        auto_to_device: bool = True,
        eval_mode: bool = True,
    ):
        """
        保留 from_config 写法，兼容旧代码。
        """

        return cls(
            unet_config=None,
            model_path=None,
            config_path=config_path,
            project_root=project_root,
            device=device,
            use_float16=use_float16,
            auto_to_device=auto_to_device,
            eval_mode=eval_mode,
        )

    def to(self, device):
        """
        手动切换 device。
        """

        self.device = torch.device(device)

        self.model.to(
            self.device,
        )

        self.pe.to(
            self.device,
        )

        return self

    def half(self):
        """
        手动切换 fp16。
        """

        self.use_float16 = True
        self.model = self.model.half()
        self.pe = self.pe.half()

        return self

    def float(self):
        """
        手动切换 fp32。
        """

        self.use_float16 = False
        self.model = self.model.float()
        self.pe = self.pe.float()

        return self

    def eval(self):
        self.model.eval()
        self.pe.eval()
        return self

    def train(self):
        self.model.train()
        self.pe.train()
        return self

    @property
    def dtype(self):
        return next(
            self.model.parameters()
        ).dtype

    def encode_audio_feature(
        self,
        whisper_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 whisper feature 加位置编码。

        输入:
            whisper_batch: [B, T, 384]

        输出:
            audio_feature_batch: [B, T, 384]
        """

        whisper_batch = whisper_batch.to(
            device=self.device,
            dtype=self.dtype,
        )

        return self.pe(
            whisper_batch,
        )

    def forward(
        self,
        latent_batch: torch.Tensor,
        whisper_batch: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MuseTalk UNet 推理封装。

        输入:
            latent_batch:
                [B, 8, 32, 32]

            whisper_batch:
                [B, T, 384]

            timesteps:
                默认 tensor([0])

        输出:
            pred_latents:
                [B, 4, 32, 32]
        """

        if timesteps is None:
            timesteps = torch.tensor(
                [0],
                device=self.device,
            )

        latent_batch = latent_batch.to(
            device=self.device,
            dtype=self.dtype,
        )

        timesteps = timesteps.to(
            device=self.device,
        )

        audio_feature_batch = self.encode_audio_feature(
            whisper_batch,
        )

        def _forward_unet():
            return self.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample

        pred_latents = self._run_without_cudnn(_forward_unet)

        return pred_latents

    def __call__(
        self,
        latent_batch: torch.Tensor,
        whisper_batch: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            latent_batch=latent_batch,
            whisper_batch=whisper_batch,
            timesteps=timesteps,
        )


if __name__ == "__main__":
    unet = UNet()

    print("UNet load success")
    print("device:", unet.device)
    print("dtype:", unet.dtype)
    print("model_path:", unet.model_path)
    print("unet_config:", unet.unet_config)