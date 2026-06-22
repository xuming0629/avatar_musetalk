import torch
import torch.nn as nn
import math
import json
from diffusers import UNet2DConditionModel
import os


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class UNet:
    """通用 UNet 模型加载类（支持 CUDA / NPU / CPU）"""
    def __init__(self, 
                 unet_config,
                 model_path,
                 use_float16=False):

        # ------------------- 检查硬件环境 -------------------
        self.device = self._select_device()

        # ------------------- 加载配置 -------------------
        with open(unet_config, 'r') as f:
            unet_config = json.load(f)

        self.model = UNet2DConditionModel(**unet_config)
        self.pe = PositionalEncoding(d_model=384)

        # ------------------- 加载权重 -------------------
        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights, strict=False)

        # ------------------- 精度设置 -------------------
        if use_float16:
            self.model = self.model.half()

        # ------------------- 转设备 + 推理模式 -------------------
        self.model.to(self.device)
        self.model.eval()

        print(f"[✅] 模型加载成功，运行设备: {self.device}")

    def _select_device(self):
        """自动选择 CUDA / NPU / CPU"""
        # 优先级：NPU > CUDA > CPU
        try:
            import torch_npu
            if torch.npu.is_available():
                print("[INFO] 检测到华为昇腾 NPU")
                return torch.device("npu:0")
        except ImportError:
            pass

        if torch.cuda.is_available():
            print("[INFO] 检测到 NVIDIA CUDA GPU")
            return torch.device("cuda")

        print("[INFO] 使用 CPU 模式")
        return torch.device("cpu")


from vae import VAE
#if __name__ == "__main__":
#    # 示例
#    unet = UNet(
#        unet_config="/data/models/digitalhuman_models/musetalk/musetalk.json",
#        model_path="/data/models/digitalhuman_models/musetalk/pytorch_model.bin",
#        use_float16=False
#    )
#    device = torch.device("npu:0")
#
#    timesteps = torch.tensor([0], device=device)
#    vae_mode_path = "/data/models/digitalhuman_models/sd-vae-ft-mse/"
#    vae = VAE(model_path=vae_mode_path, use_float16=True)
#    
#    img_path = "3456.png"
#    latent_batch = vae.get_latents_for_unet(img_path)
#    print(latent_batch.shape)
#    print("sucessfuly .......")
#
#    #pred_latents = unet.model(latent_batch,
#    #                          timesteps,
#    #                          encoder_hidden_states=audio_feature_batch).sample
#
#    print("===============================")
#if __name__ == "__main__":
#    # 1️⃣ 初始化 UNet
#    unet = UNet(
#        unet_config="/data/models/digitalhuman_models/musetalk/musetalk.json",
#        model_path="/data/models/digitalhuman_models/musetalk/pytorch_model.bin",
#        use_float16=True
#    )
#
#    # 2️⃣ 初始化 VAE
#    vae_mode_path = "/data/models/digitalhuman_models/sd-vae-ft-mse/"
#    vae = VAE(model_path=vae_mode_path, use_float16=True)
#   
#    device = torch.device("npu:0")
#
#    # 3️⃣ 生成 latent
#    img_path = "3456.png"
#    latent_batch = vae.get_latents_for_unet(img_path).to(device)
#    print(latent_batch.type)
#    print("Latent shape:", latent_batch.shape)
#    
#    # 4️⃣ 生成模拟音频特征
#    audio_feature_batch = torch.randn(1, 50, 384, device=device)  # 假设音频 embedding
#    print(audio_feature_batch.type)
#    timesteps = torch.tensor([0], device=device)
#    print(timesteps.type)
#
#    # 5️⃣ 执行 U-Net 推理
#    with torch.no_grad():
#        pred_latents = unet.model(
#            latent_batch,
#            timesteps,
#            encoder_hidden_states=audio_feature_batch
#        ).sample
#
#    print("预测输出形状:", pred_latents.shape)
#    print("✅ 成功完成推理！")
if __name__ == "__main__":
    # ---------------- 1. 初始化 UNet ----------------
    unet = UNet(
        unet_config="/data/models/digitalhuman_models/musetalk/musetalk.json",
        model_path="/data/models/digitalhuman_models/musetalk/pytorch_model.bin",
        use_float16=True  # 使用 fp16
    )

    # ---------------- 2. 初始化 VAE ----------------
    vae_mode_path = "/data/models/digitalhuman_models/sd-vae-ft-mse/"
    vae = VAE(model_path=vae_mode_path, use_float16=True)

    # ---------------- 3. 设备选择 ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_npu
            device = torch.device("npu:0")
        except ImportError:
            device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------- 4. 生成 latent ----------------
    img_path = "3456.png"
    latent_batch = vae.get_latents_for_unet(img_path).to(device)
    # 保证 latent_batch 为 fp16
    if unet.model.dtype == torch.float16:
        latent_batch = latent_batch.half()
    else:
        latent_batch = latent_batch.float()
    print("latent_batch dtype:", latent_batch.dtype)
    print("Latent shape:", latent_batch.shape)

    # ---------------- 5. 模拟音频特征 ----------------
    audio_feature_batch = torch.randn(1, 50, 384, device=device)
    # 同样转为 fp16
    if unet.model.dtype == torch.float16:
        audio_feature_batch = audio_feature_batch.half()
    else:
        audio_feature_batch = audio_feature_batch.float()
    print("audio_feature_batch dtype:", audio_feature_batch.dtype)

    # timesteps 一般用 long/int64
    timesteps = torch.tensor([0], device=device, dtype=torch.long)
    print("timesteps dtype:", timesteps.dtype)

    # ---------------- 6. U-Net 推理 ----------------
    with torch.no_grad():
        pred_latents = unet.model(
            latent_batch,
            timesteps,
            encoder_hidden_states=audio_feature_batch
        ).sample

    print("pred_latents dtype:", pred_latents.dtype)
    print("pred_latents shape:", pred_latents.shape)
    print("[INFO] 推理成功 ✅")
