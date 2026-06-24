# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# # @FileName      : sfd_detector.py
# # @Time          : 2026-06-23 22:56:44
# # @Author        : XuMing
# # @Email         : 920972751@qq.com
# # @description   : S3FD face detector with config support
# # @Company       : 2026 XuMing. All Rights Reserved.
# """




# import os
# from typing import Optional

# import torch
# from torch.utils.model_zoo import load_url

# from src.nets.common.config import (
#     load_yaml,
#     resolve_path,
#     get_device,
# )

# from src.nets.face.s3fd.face_detect import FaceDetector

# from .s3fd_net import s3fd
# from .bbox import *
# from .detect import *


# models_urls = {
#     "s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
# }


# def get_s3fd_path(
#         config_path: str = "configs/musetalk_v15.yaml",
#         project_root: Optional[str] = None,
# ) -> str:
#     cfg = load_yaml(config_path)

#     model_path = cfg["models"]["s3fd"]["path"]

#     model_path = resolve_path(
#         model_path,
#         project_root,
#     )

#     return model_path


# class SFDDetector(FaceDetector):

#     def __init__(
#             self,
#             device,
#             path_to_detector: Optional[str] = None,
#             verbose: bool = False,
#             auto_download: bool = False,
#     ):
#         super().__init__(
#             device,
#             verbose,
#         )

#         self.device = str(device)

#         if path_to_detector is None:
#             raise ValueError(
#                 "path_to_detector is None. "
#                 "Please pass model path or use SFDDetector.from_config()."
#             )

#         if not os.path.isfile(path_to_detector):
#             if auto_download:
#                 print(
#                     f"[SFDDetector] model not found, download from: {models_urls['s3fd']}"
#                 )
#                 model_weights = load_url(
#                     models_urls["s3fd"],
#                     map_location="cpu",
#                 )
#             else:
#                 raise FileNotFoundError(
#                     f"S3FD detector weight not found: {path_to_detector}"
#                 )
#         else:
#             model_weights = torch.load(
#                 path_to_detector,
#                 map_location="cpu",
#                 weights_only=True,
#             )

#         self.face_detector = s3fd()
#         self.face_detector.load_state_dict(
#             model_weights,
#             strict=True,
#         )

#         self.face_detector.to(
#             self.device,
#         )

#         self.face_detector.eval()

#         print(
#             f"[SFDDetector] load weight success: {path_to_detector}"
#         )

#     @classmethod
#     def from_config(
#             cls,
#             config_path: str = "configs/musetalk_v15.yaml",
#             project_root: Optional[str] = None,
#             verbose: bool = False,
#             auto_download: bool = False,
#     ):
#         device = get_device(
#             config_path=config_path,
#         )

#         model_path = get_s3fd_path(
#             config_path=config_path,
#             project_root=project_root,
#         )

#         return cls(
#             device=device,
#             path_to_detector=model_path,
#             verbose=verbose,
#             auto_download=auto_download,
#         )

#     def detect_from_image(self, tensor_or_path):
#         image = self.tensor_or_path_to_ndarray(
#             tensor_or_path,
#         )

#         bboxlist = detect(
#             self.face_detector,
#             image,
#             device=self.device,
#         )

#         keep = nms(
#             bboxlist,
#             0.3,
#         )

#         bboxlist = bboxlist[
#             keep,
#             :,
#         ]

#         bboxlist = [
#             x for x in bboxlist
#             if x[-1] > 0.5
#         ]

#         return bboxlist

#     def detect_from_batch(self, images):
#         bboxlists = batch_detect(
#             self.face_detector,
#             images,
#             device=self.device,
#         )

#         keeps = [
#             nms(
#                 bboxlists[:, i, :],
#                 0.3,
#             )
#             for i in range(bboxlists.shape[1])
#         ]

#         bboxlists = [
#             bboxlists[
#                 keep,
#                 i,
#                 :,
#             ]
#             for i, keep in enumerate(keeps)
#         ]

#         bboxlists = [
#             [
#                 x for x in bboxlist
#                 if x[-1] > 0.5
#             ]
#             for bboxlist in bboxlists
#         ]

#         return bboxlists

#     @property
#     def reference_scale(self):
#         return 195

#     @property
#     def reference_x_shift(self):
#         return 0

#     @property
#     def reference_y_shift(self):
#         return 0


# if __name__ == "__main__":
#     detector = SFDDetector.from_config(
#         "configs/musetalk_v15.yaml",
#     )

#     img_path = "assets/sit.jpeg"

#     bboxes = detector.detect_from_image(
#         img_path,
#     )

#     print("bboxes:")
#     for bbox in bboxes:
#         print(bbox)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : sfd_detector.py
# @Time          : 2026-06-23 22:56:44
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : S3FD face detector with config support
# @Company       : 2026 XuMing. All Rights Reserved.
"""

import os
from typing import Optional

import torch
from torch.utils.model_zoo import load_url

from src.nets.common.config import (
    load_yaml,
    resolve_path,
    get_device,
)

from src.nets.face.s3fd.face_detect import FaceDetector

from .s3fd_net import s3fd
from .bbox import *
from .detect import *


DEFAULT_CONFIG_PATH = "configs/musetalk_v15.yaml"


models_urls = {
    "s3fd": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
}


def get_s3fd_path(
    config_path: str = DEFAULT_CONFIG_PATH,
    project_root: Optional[str] = None,
) -> str:
    cfg = load_yaml(config_path)

    model_path = cfg["models"]["s3fd"]["path"]

    model_path = resolve_path(
        model_path,
        project_root,
    )

    return model_path


class SFDDetector(FaceDetector):
    """
    S3FD 人脸检测器。

    默认支持直接声明：

        detector = SFDDetector()

    等价于：

        detector = SFDDetector.from_config("configs/musetalk_v15.yaml")
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        device=None,
        path_to_detector: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        """
        参数:
            config_path:
                默认配置文件路径。

            project_root:
                项目根目录，用于 resolve_path。

            device:
                可选。如果不传，自动从 config 读取。

            path_to_detector:
                可选。如果不传，自动从 config 中读取 models.s3fd.path。

            verbose:
                是否打印 FaceDetector 内部日志。

            auto_download:
                如果模型不存在，是否自动下载。
        """

        if device is None:
            device = get_device(
                config_path=config_path,
            )

        if path_to_detector is None:
            path_to_detector = get_s3fd_path(
                config_path=config_path,
                project_root=project_root,
            )

        super().__init__(
            device,
            verbose,
        )

        self.config_path = config_path
        self.project_root = project_root
        self.device = str(device)
        self.path_to_detector = path_to_detector

        if not os.path.isfile(path_to_detector):
            if auto_download:
                print(
                    f"[SFDDetector] model not found, download from: {models_urls['s3fd']}"
                )
                model_weights = load_url(
                    models_urls["s3fd"],
                    map_location="cpu",
                )
            else:
                raise FileNotFoundError(
                    f"S3FD detector weight not found: {path_to_detector}"
                )
        else:
            try:
                model_weights = torch.load(
                    path_to_detector,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # 兼容旧版本 PyTorch，不支持 weights_only 参数
                model_weights = torch.load(
                    path_to_detector,
                    map_location="cpu",
                )

        self.face_detector = s3fd()

        self.face_detector.load_state_dict(
            model_weights,
            strict=True,
        )

        self.face_detector.to(
            self.device,
        )

        self.face_detector.eval()

        print(
            f"[SFDDetector] load weight success: {path_to_detector}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        project_root: Optional[str] = None,
        verbose: bool = False,
        auto_download: bool = False,
    ):
        """
        保留 from_config 写法，兼容旧代码。
        """

        return cls(
            config_path=config_path,
            project_root=project_root,
            device=None,
            path_to_detector=None,
            verbose=verbose,
            auto_download=auto_download,
        )

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(
            tensor_or_path,
        )

        bboxlist = detect(
            self.face_detector,
            image,
            device=self.device,
        )

        keep = nms(
            bboxlist,
            0.3,
        )

        bboxlist = bboxlist[
            keep,
            :,
        ]

        bboxlist = [
            x for x in bboxlist
            if x[-1] > 0.5
        ]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(
            self.face_detector,
            images,
            device=self.device,
        )

        keeps = [
            nms(
                bboxlists[:, i, :],
                0.3,
            )
            for i in range(bboxlists.shape[1])
        ]

        bboxlists = [
            bboxlists[
                keep,
                i,
                :,
            ]
            for i, keep in enumerate(keeps)
        ]

        bboxlists = [
            [
                x for x in bboxlist
                if x[-1] > 0.5
            ]
            for bboxlist in bboxlists
        ]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0


