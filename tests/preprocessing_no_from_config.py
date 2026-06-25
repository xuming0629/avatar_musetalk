#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : preprocessing.py
# @description   : DWPose + S3FD/FaceAlignment 人脸 bbox 提取
#
# 说明：
# - 不使用 from_config()
# - 全局 get_landmark_and_bbox() 会自动创建 RTMPose() + FaceAlignment()
# - 兼容原 MuseTalk 调用方式：
#       from src.utils.preprocessing import get_landmark_and_bbox, get_bbox_range, coord_placeholder
"""

import os
import pickle
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
from tqdm import tqdm

from src.nets.face.alignment.face_alignment import FaceAlignment
from src.nets.face.dwpose.dwpose import RTMPose


coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def read_imgs(
    img_list: List[str],
) -> List[np.ndarray]:
    """读取图片列表，返回 OpenCV BGR 图像。"""
    frames = []

    print("reading images...")

    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)

        if frame is None:
            raise FileNotFoundError(
                f"读取图片失败: {img_path}"
            )

        frames.append(frame)

    return frames


def clip_bbox(
    bbox,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """将 bbox 裁剪到图像范围内。"""
    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(int(x1), width - 1),
    )

    y1 = max(
        0,
        min(int(y1), height - 1),
    )

    x2 = max(
        0,
        min(int(x2), width),
    )

    y2 = max(
        0,
        min(int(y2), height),
    )

    return x1, y1, x2, y2


def is_valid_bbox(
    bbox,
    min_size: int = 2,
) -> bool:
    """检查 bbox 是否有效。"""
    if bbox is None:
        return False

    try:
        x1, y1, x2, y2 = bbox[:4]
    except Exception:
        return False

    if x2 - x1 < min_size:
        return False

    if y2 - y1 < min_size:
        return False

    return True


def normalize_face_bbox(
    bboxes: Any,
):
    """
    兼容不同人脸检测器返回格式。

    支持：
    - None
    - [x1, y1, x2, y2]
    - [x1, y1, x2, y2, score]
    - np.ndarray shape=(4,)
    - np.ndarray shape=(5,)
    - np.ndarray shape=(N, 4)
    - np.ndarray shape=(N, 5)
    - list[np.ndarray/list]
    """
    if bboxes is None:
        return None

    if isinstance(bboxes, np.ndarray):
        arr = bboxes

        if arr.size == 0:
            return None

        arr = np.asarray(arr)

        if arr.ndim == 1:
            if arr.shape[0] >= 4:
                return tuple(arr[:4].astype(float).tolist())
            return None

        if arr.ndim == 2:
            if arr.shape[0] == 0 or arr.shape[1] < 4:
                return None

            # 如果有 score，优先选 score 最大的人脸
            if arr.shape[1] >= 5:
                best_idx = int(np.argmax(arr[:, 4]))
                return tuple(arr[best_idx, :4].astype(float).tolist())

            # 否则选第一个
            return tuple(arr[0, :4].astype(float).tolist())

        return None

    if isinstance(bboxes, (list, tuple)):
        if len(bboxes) == 0:
            return None

        # [x1, y1, x2, y2] / [x1, y1, x2, y2, score]
        if len(bboxes) >= 4 and all(
            isinstance(v, (int, float, np.integer, np.floating))
            for v in bboxes[:4]
        ):
            return tuple(float(v) for v in bboxes[:4])

        # list of boxes
        candidates = []
        for item in bboxes:
            box = normalize_face_bbox(item)
            if box is not None:
                candidates.append(box)

        if len(candidates) == 0:
            return None

        # 选面积最大
        def area(box):
            x1, y1, x2, y2 = box
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return max(candidates, key=area)

    return None


class LandmarkBBoxExtractor:
    """
    人脸关键点和 bbox 提取器。

    依赖：
        1. RTMPose / DWPose：提取 wholebody keypoints
        2. FaceAlignment/S3FD：做人脸检测

    注意：
        这个版本不提供 from_config()，也不调用 from_config()。
        需要外部直接传入已经初始化好的 pose_model 和 face_detector，
        或者使用 create_default_extractor()。
    """

    def __init__(
        self,
        pose_model: Optional[RTMPose] = None,
        face_detector: Optional[FaceAlignment] = None,
    ):
        # 不使用 from_config，直接默认构造
        self.pose_model = pose_model if pose_model is not None else RTMPose()
        self.face_detector = (
            face_detector if face_detector is not None else FaceAlignment()
        )

    def _get_face_landmark(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        获取单帧人脸关键点。

        RTMPose wholebody:
            keypoints[23:91] 为 face landmarks
        """
        keypoints, scores = self.pose_model.predict(
            frame,
            out_bbox=None,
        )

        if keypoints is None or len(keypoints) == 0:
            return None

        keypoints = np.asarray(keypoints)

        if keypoints.ndim != 3:
            raise ValueError(
                f"RTMPose keypoints 维度异常: {keypoints.shape}"
            )

        if keypoints.shape[1] < 91:
            raise ValueError(
                f"RTMPose keypoints 数量不足: {keypoints.shape}"
            )

        face_land_mark = keypoints[0][23:91]

        return face_land_mark.astype(
            np.int32
        )

    def _get_face_bbox(
        self,
        frame: np.ndarray,
    ):
        """使用 FaceAlignment/S3FD 获取单帧人脸 bbox。"""
        if not hasattr(self.face_detector, "get_detections_for_batch"):
            raise AttributeError(
                "face_detector 缺少 get_detections_for_batch 方法，"
                "请检查 FaceAlignment 或 SFDDetector 实现。"
            )

        bboxes = self.face_detector.get_detections_for_batch(
            np.asarray([frame])
        )

        bbox = normalize_face_bbox(bboxes)

        if bbox is None:
            return None

        return bbox

    def _build_bbox_from_landmark(
        self,
        face_land_mark: np.ndarray,
        fallback_bbox,
        image_shape,
        upperbondrange: int = 0,
    ):
        """
        根据 face landmark 构造 MuseTalk 裁剪 bbox。

        如果 landmark 构造失败，则回退到 face detector bbox。
        """
        height, width = image_shape[:2]

        range_minus = (
            face_land_mark[30]
            - face_land_mark[29]
        )[1]

        range_plus = (
            face_land_mark[29]
            - face_land_mark[28]
        )[1]

        half_face_coord = face_land_mark[29].copy()

        if upperbondrange != 0:
            half_face_coord[1] = (
                half_face_coord[1]
                + upperbondrange
            )

        half_face_dist = (
            np.max(face_land_mark[:, 1])
            - half_face_coord[1]
        )

        upper_bond = max(
            0,
            half_face_coord[1] - half_face_dist,
        )

        landmark_bbox = (
            int(np.min(face_land_mark[:, 0])),
            int(upper_bond),
            int(np.max(face_land_mark[:, 0])),
            int(np.max(face_land_mark[:, 1])),
        )

        if is_valid_bbox(landmark_bbox):
            bbox = landmark_bbox
        elif is_valid_bbox(fallback_bbox):
            bbox = fallback_bbox
            print(
                "error landmark bbox, fallback to face detector:",
                bbox,
            )
        else:
            return coord_placeholder, range_minus, range_plus

        bbox = clip_bbox(
            bbox,
            width=width,
            height=height,
        )

        if not is_valid_bbox(bbox):
            return coord_placeholder, range_minus, range_plus

        return bbox, range_minus, range_plus

    def get_bbox_range(
        self,
        img_list: List[str],
        upperbondrange: int = 0,
    ) -> str:
        """
        获取 bbox_shift 推荐范围。
        """
        frames = read_imgs(
            img_list
        )

        average_range_minus = []
        average_range_plus = []

        print(
            "get key_landmark and face bounding boxes "
            f"with bbox_shift: {upperbondrange}"
        )

        for frame in tqdm(frames):
            face_land_mark = self._get_face_landmark(
                frame
            )

            face_bbox = self._get_face_bbox(
                frame
            )

            if face_land_mark is None or face_bbox is None:
                continue

            _, range_minus, range_plus = self._build_bbox_from_landmark(
                face_land_mark=face_land_mark,
                fallback_bbox=face_bbox,
                image_shape=frame.shape,
                upperbondrange=upperbondrange,
            )

            average_range_minus.append(
                range_minus
            )

            average_range_plus.append(
                range_plus
            )

        if len(average_range_minus) == 0:
            return (
                f"Total frame:「{len(frames)}」 "
                "No valid face detected."
            )

        return (
            f"Total frame:「{len(frames)}」 "
            f"Manually adjust range : "
            f"[ -{int(np.mean(average_range_minus))}"
            f"~{int(np.mean(average_range_plus))} ] , "
            f"the current value: {upperbondrange}"
        )

    def get_landmark_and_bbox(
        self,
        img_list: List[str],
        upperbondrange: int = 0,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        获取每一帧的人脸裁剪 bbox。

        Args:
            img_list: 图片路径列表
            upperbondrange: bbox_shift，正数向下，负数向上

        Returns:
            coords_list: bbox 列表
            frames: 原始图像列表
        """
        frames = read_imgs(
            img_list
        )

        coords_list = []
        average_range_minus = []
        average_range_plus = []

        print(
            "get key_landmark and face bounding boxes "
            f"with bbox_shift: {upperbondrange}"
        )

        for frame in tqdm(frames):
            face_land_mark = self._get_face_landmark(
                frame
            )

            face_bbox = self._get_face_bbox(
                frame
            )

            if face_land_mark is None or face_bbox is None:
                coords_list.append(
                    coord_placeholder
                )
                continue

            bbox, range_minus, range_plus = self._build_bbox_from_landmark(
                face_land_mark=face_land_mark,
                fallback_bbox=face_bbox,
                image_shape=frame.shape,
                upperbondrange=upperbondrange,
            )

            coords_list.append(
                bbox
            )

            if bbox != coord_placeholder:
                average_range_minus.append(
                    range_minus
                )

                average_range_plus.append(
                    range_plus
                )

        if len(average_range_minus) > 0:
            print(
                "********************************************\n"
                "bbox_shift parameter adjustment\n"
                "********************************************"
            )

            print(
                f"Total frame:「{len(frames)}」 "
                f"Manually adjust range : "
                f"[ -{int(np.mean(average_range_minus))}"
                f"~{int(np.mean(average_range_plus))} ] , "
                f"the current value: {upperbondrange}"
            )

            print(
                "****************************************************************"
            )

        return coords_list, frames


# ============================================================
# 全局 extractor 缓存
# 兼容原 MuseTalk 写法：
# from src.utils.preprocessing import get_landmark_and_bbox, get_bbox_range
# ============================================================

_GLOBAL_EXTRACTOR = None


def create_default_extractor() -> LandmarkBBoxExtractor:
    """
    创建默认 LandmarkBBoxExtractor。

    这里不使用 from_config，直接：
        RTMPose()
        FaceAlignment()
    """
    pose_model = RTMPose()
    face_detector = FaceAlignment()

    return LandmarkBBoxExtractor(
        pose_model=pose_model,
        face_detector=face_detector,
    )


def get_global_extractor() -> LandmarkBBoxExtractor:
    """
    获取全局 LandmarkBBoxExtractor，只初始化一次。

    注意：
        不接受 config_path，不调用 from_config。
    """
    global _GLOBAL_EXTRACTOR

    if _GLOBAL_EXTRACTOR is None:
        _GLOBAL_EXTRACTOR = create_default_extractor()

    return _GLOBAL_EXTRACTOR


def get_landmark_and_bbox(
    img_list: List[str],
    upperbondrange: int = 0,
    config_path: Optional[str] = None,
):
    """
    模块级兼容函数。

    原 inference.py 可以直接这样调用：
        coord_list, frame_list = get_landmark_and_bbox(img_list, bbox_shift)

    config_path 参数仅保留兼容旧调用，不再使用。
    """
    extractor = get_global_extractor()

    return extractor.get_landmark_and_bbox(
        img_list=img_list,
        upperbondrange=upperbondrange,
    )


def get_bbox_range(
    img_list: List[str],
    upperbondrange: int = 0,
    config_path: Optional[str] = None,
):
    """
    模块级兼容函数。

    config_path 参数仅保留兼容旧调用，不再使用。
    """
    extractor = get_global_extractor()

    return extractor.get_bbox_range(
        img_list=img_list,
        upperbondrange=upperbondrange,
    )


if __name__ == "__main__":
    extractor = create_default_extractor()

    img_list = [
        "./results/lyria/00000.png",
        "./results/lyria/00001.png",
        "./results/lyria/00002.png",
        "./results/lyria/00003.png",
    ]

    crop_coord_path = "./coord_face.pkl"

    coords_list, full_frames = extractor.get_landmark_and_bbox(
        img_list
    )

    print(coords_list)

    with open(
        crop_coord_path,
        "wb",
    ) as f:
        pickle.dump(
            coords_list,
            f,
        )

    for bbox, frame in zip(
        coords_list,
        full_frames,
    ):
        if bbox == coord_placeholder:
            continue

        x1, y1, x2, y2 = bbox

        crop_frame = frame[
            y1:y2,
            x1:x2,
        ]

        print(
            "Cropped shape",
            crop_frame.shape,
        )

    print(
        coords_list
    )
