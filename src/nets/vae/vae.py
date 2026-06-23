from diffusers import AutoencoderKL
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from PIL import Image

# 如果安装了 torch_npu，则导入
try:
    import torch_npu
    has_npu = torch.npu.is_available()
except ImportError:
    has_npu = False

#from config import model_config


class VAE:
    """
    VAE (Variational Autoencoder) class for image processing on Ascend NPU.
    """

    def __init__(self, model_path, resized_img=256, use_float16=False):
        """
        Initialize the VAE instance.

        :param model_path: Path to pretrained VAE model.
        :param resized_img: Target image size.
        :param use_float16: Whether to use float16 precision.
        """
        self.model_path = model_path
        self.vae = AutoencoderKL.from_pretrained(self.model_path)

        # ✅ 优先使用 NPU
        if has_npu:
            self.device = torch.device("npu")
            print("[INFO] Using Ascend NPU for inference.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[INFO] Using CUDA GPU for inference.")
        else:
            self.device = torch.device("cpu")
            print("[WARN] Using CPU for inference, may be slow.")

        # ✅ 模型精度
        if use_float16:
            self.vae = self.vae.half()
            self._use_float16 = True
        else:
            self._use_float16 = False

        self.vae.to(self.device)
        self.vae.eval()

        self.scaling_factor = self.vae.config.scaling_factor
        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])
        self._resized_img = resized_img
        self._mask_tensor = self.get_mask_tensor()

    def get_mask_tensor(self):
        """Create a half-mask tensor."""
        mask_tensor = torch.zeros((self._resized_img, self._resized_img))
        mask_tensor[:self._resized_img // 2, :] = 1
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1
        return mask_tensor

    def preprocess_img(self, img_name, half_mask=False):
        """Preprocess an image for the VAE."""
        window = []
        if isinstance(img_name, str):
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self._resized_img, self._resized_img),
                             interpolation=cv2.INTER_LANCZOS4)
            window.append(img)
        else:
            img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
            window.append(img)

        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        x = torch.squeeze(torch.FloatTensor(x))

        if half_mask:
            x = x * (self._mask_tensor > 0.5)
        x = self.transform(x)

        x = x.unsqueeze(0)  # [1, 3, 256, 256]
        x = x.to(self.device, dtype=torch.float16 if self._use_float16 else torch.float32)

        return x

    def encode_latents(self, image):
        """Encode image to latent variables."""
        with torch.no_grad():
            init_latent_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents

    def decode_latents(self, latents):
        """Decode latent variables back to image."""
        latents = (1 / self.scaling_factor) * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[..., ::-1]  # RGB → BGR
        return image

    def get_latents_for_unet(self, img):
        """Prepare latents for UNet input."""
        ref_image = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref_image)

        ref_image = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref_image)

        latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
        return latent_model_input


if __name__ == "__main__":
    vae_mode_path = "/data/models/digitalhuman_models/sd-vae-ft-mse/"
    vae = VAE(model_path=vae_mode_path, use_float16=True)


    #crop_imgs_path = "./results/sun001_crop/"
    #latents_out_path = "./results/latents/"
    #os.makedirs(latents_out_path, exist_ok=True)

    #files = sorted([f for f in os.listdir(crop_imgs_path) if f.endswith(".png")])

    #for file in files:
    #    img_path = os.path.join(crop_imgs_path, file)
    img_path = "3456.png"
    latents = vae.get_latents_for_unet(img_path)
    print(latents)
    #    print(img_path, "latents", latents.size())
    #    # torch.save(latents, os.path.join(latents_out_path, file.replace(".png", ".pt")))

