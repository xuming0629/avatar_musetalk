import os
import sys
import json
import math

import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        max_len: int = 5000,
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        _, seq_len, _ = x.size()

        pe = self.pe[
            :,
            :seq_len,
            :
        ]

        x = x + pe.to(
            x.device
        )

        return x


class UNet:
    def __init__(
        self,
        unet_config: str,
        model_path: str,
        device: str = "cpu",
        use_float16: bool = False,
    ):
        self.unet_config = unet_config
        self.model_path = model_path
        self.device = torch.device(device)
        self.use_float16 = use_float16

        with open(
            self.unet_config,
            "r",
            encoding="utf-8"
        ) as f:
            unet_cfg = json.load(f)

        self.model = UNet2DConditionModel(
            **unet_cfg
        )

        self.pe = PositionalEncoding(
            d_model=384
        )

        weights = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            weights
        )

        if self.use_float16:
            self.model = self.model.half()

        self.model.to(
            self.device
        )

        self.pe.to(
            self.device
        )

        self.model.eval()
        self.pe.eval()

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/musetalk_v15.yaml",
    ):
        from src.nets.common.config import load_yaml, get_device

        cfg = load_yaml(
            config_path
        )

        model_cfg = (
            cfg.get("models", {})
               .get("musetalk", {})
        )

        model_path = model_cfg.get("path")
        unet_config = model_cfg.get("cfg")

        if model_path is None:
            raise ValueError(
                "Config error: models.musetalk.path not found"
            )

        if unet_config is None:
            raise ValueError(
                "Config error: models.musetalk.cfg not found"
            )

        use_float16 = model_cfg.get(
            "use_float16",
            False
        )

        return cls(
            unet_config=unet_config,
            model_path=model_path,
            device=get_device(config_path),
            use_float16=use_float16,
        )


if __name__ == "__main__":

    unet = UNet.from_config(
        "configs/musetalk_v15.yaml"
    )

    print("UNet load success")
    print("device:", unet.device)