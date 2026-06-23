#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.nets.common.config import load_yaml, resolve_path, get_device
from src.nets.face.parsing.face_parse_bisenet import BiSeNet


class FaceParsing:
    def __init__(
            self,
            config_path="configs/musetalk_v15.yaml",
            project_root=None,
    ):
        self.config_path = config_path
        self.project_root = project_root

        self.cfg = load_yaml(config_path)
        self.device = get_device(config_path)

        print(f"[FaceParsing] device: {self.device}")

        self.resnet_path = self._get_resnet_path()
        self.model_path = self._get_face_parse_path()
        self.num_classes = self._get_num_classes()

        self.net = self.model_init()
        self.preprocess = self.image_preprocess()

    def _get_resnet_path(self):
        path = self.cfg["models"]["resnet18"]["path"]
        path = resolve_path(path, self.project_root)
        return path

    def _get_face_parse_path(self):
        path = self.cfg["models"]["face-parse-bisent"]["path"]
        path = resolve_path(path, self.project_root)
        return path

    def _get_num_classes(self):
        return self.cfg["models"]["face-parse-bisent"].get(
            "num_classes",
            19,
        )

    def model_init(self):
        net = BiSeNet(
            resnet_path=self.resnet_path,
            n_classes=self.num_classes,
        )

        state_dict = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

        net.load_state_dict(
            state_dict,
            strict=False,
        )

        net.to(self.device)
        net.eval()

        print(f"[FaceParsing] load BiSeNet: {self.model_path}")
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

    def __call__(
            self,
            image,
            size=(512, 512)
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

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
            

    def save(
            self,
            image,
            save_path,
            size=(512, 512),
            mode="raw",
    ):
        mask = self(
            image=image,
            size=size
        )
        mask.save(save_path)
        return save_path


if __name__ == "__main__":
    fp = FaceParsing(
        config_path="configs/musetalk_v15.yaml",
    )

    save_path = fp.save(
        image="./assets/face_0.png",
        save_path="./assets/outputs/res.png",
        mode="raw",
    )

    print(f"[FaceParsing] save: {save_path}")