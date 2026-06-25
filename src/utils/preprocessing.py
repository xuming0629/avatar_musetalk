#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : preprocessing.py
# @description   : DWPose + S3FD 人脸 bbox 提取
"""

import os
import pickle
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.nets.face.alignment.face_alignment import FaceAlignment
from src.nets.face.dwpose.dwpose import RTMPose


coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def read_imgs(
    img_list: List[str],
) -> List[np.ndarray]:
    """读取图片列表。"""
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


class LandmarkBBoxExtractor:
    """
    人脸关键点和 bbox 提取器。

    依赖：
        1. RTMPose / DWPose：提取 wholebody keypoints
        2. S3FD：做人脸检测
    """

    def __init__(
        self,
        pose_model: RTMPose,
        face_detector: FaceAlignment,
    ):
        self.pose_model = pose_model
        self.face_detector = face_detector

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/musetalk_v15.yaml",
    ):
        pose_model = RTMPose.from_config(
            config_path
        )

        face_detector = FaceAlignment.from_config(
            config_path
        )

        return cls(
            pose_model=pose_model,
            face_detector=face_detector,
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
        """使用 S3FD 获取单帧人脸 bbox。"""
        bboxes = self.face_detector.get_detections_for_batch(
            np.asarray([frame])
        )

        if bboxes is None or len(bboxes) == 0:
            return None

        return bboxes[0]

    def _build_bbox_from_landmark(
        self,
        face_land_mark: np.ndarray,
        fallback_bbox,
        image_shape,
        upperbondrange: int = 0,
    ):
        """
        根据 face landmark 构造 MuseTalk 裁剪 bbox。

        如果 landmark 构造失败，则回退到 S3FD bbox。
        """
        height, width = image_shape[:2]

        half_face_coord = face_land_mark[29].copy()

        range_minus = (
            face_land_mark[30]
            - face_land_mark[29]
        )[1]

        range_plus = (
            face_land_mark[29]
            - face_land_mark[28]
        )[1]

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

        x1, y1, x2, y2 = landmark_bbox

        if (
            y2 - y1 <= 0
            or x2 - x1 <= 0
            or x1 < 0
        ):
            bbox = fallback_bbox
            print(
                "error bbox, fallback to face detector:",
                bbox,
            )
        else:
            bbox = landmark_bbox

        bbox = clip_bbox(
            bbox,
            width=width,
            height=height,
        )

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

            average_range_minus.append(
                range_minus
            )

            average_range_plus.append(
                range_plus
            )

        if len(average_range_minus) > 0:
            print(
                "********************************************"
                "bbox_shift parameter adjustment"
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


if __name__ == "__main__":

    extractor = LandmarkBBoxExtractor.from_config(
        "configs/musetalk_v15.yaml"
    )
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

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