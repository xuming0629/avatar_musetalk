#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from .face_parse_bisenet import BiSeNet



class FaceParsing:
    def __init__(self,
                 resnet_path='./models/resnet18/resnet18-5c106cde.pth',
                 model_pth='./models/face-parse-bisent/79999_iter.pth'):
        # 自动检测设备
        self.device = self.get_device()
        print(f"[INFO] Using device: {self.device}")

        # 初始化模型
        self.net = self.model_init(resnet_path, model_pth)
        self.preprocess = self.image_preprocess()

    def get_device(self):
        """自动检测 CUDA / NPU / CPU"""
        try:
            import torch_npu
            has_npu = torch.npu.is_available()
        except ImportError:
            has_npu = False

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif has_npu:
            return torch.device("npu:0")
        else:
            return torch.device("cpu")

    def model_init(self, resnet_path, model_pth):
        """加载模型"""
        net = BiSeNet(resnet_path)
        net.to(self.device)

        # 根据设备加载权重
        map_location = None if self.device.type in ["cuda", "npu"] else torch.device("cpu")
        state_dict = torch.load(model_pth, map_location=map_location)
        net.load_state_dict(state_dict)
        net.eval()
        return net

    def image_preprocess(self):
        """标准化预处理"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512)):
        """执行分割预测"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        width, height = image.size

        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            img = torch.unsqueeze(img, 0).to(self.device)

            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # 简单二值化处理
            parsing[np.where(parsing > 13)] = 0
            parsing[np.where(parsing >= 1)] = 255

        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing


if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('./assets/face_0.png')
    segmap.save('./assets/outputs/res.png')

