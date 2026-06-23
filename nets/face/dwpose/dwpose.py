#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : test_dwpose.py
# @Time          : 2026-06-22 22:49:03
# @Author        : XuMing
# @Email         : 920972751@qq.com
# @description   : 人体关键点
# @Company       : 2026 XuMing. All Rights Reserved.
"""





from typing import List, Tuple, Optional
import os
import cv2
import numpy as np
import onnxruntime as ort


from nets.common.config import (
    load_yaml,
    get_device,
)

def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center.astype(np.float32), scale.astype(np.float32)


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    shift = np.array(shift, dtype=np.float32)
    src_w = scale[0]
    dst_w, dst_h = output_size

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(
    input_size: Tuple[int, int],
    bbox_scale: np.ndarray,
    bbox_center: np.ndarray,
    img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = input_size
    warp_size = (int(w), int(h))

    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    warp_mat = get_warp_matrix(
        center=bbox_center,
        scale=bbox_scale,
        rot=0,
        output_size=(w, h),
    )

    resized_img = cv2.warpAffine(
        img,
        warp_mat,
        warp_size,
        flags=cv2.INTER_LINEAR,
    )

    return resized_img, bbox_scale


def preprocess(
    img: np.ndarray,
    out_bbox: Optional[List[List[float]]] = None,
    input_size: Tuple[int, int] = (288, 384),
    padding: float = 1.25,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    img_shape = img.shape[:2]

    if out_bbox is None or len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]

    out_img = []
    out_center = []
    out_scale = []

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    for bbox in out_bbox:
        bbox = np.array(bbox, dtype=np.float32)

        center, scale = bbox_xyxy2cs(bbox, padding=padding)
        resized_img, scale = top_down_affine(input_size, scale, center, img)

        resized_img = resized_img.astype(np.float32)
        resized_img = (resized_img - mean) / std

        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale


def inference(sess: ort.InferenceSession, imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
    all_outputs = []

    input_name = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]

    for img in imgs:
        input_data = img.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        outputs = sess.run(output_names, {input_name: input_data})
        all_outputs.append(outputs)

    return all_outputs


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K, _ = simcc_x.shape

    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)

    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]

    vals = max_val_x
    locs[vals <= 0.0] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / simcc_split_ratio
    return keypoints, scores


def postprocess(
    outputs: List[List[np.ndarray]],
    model_input_size: Tuple[int, int],
    centers: List[np.ndarray],
    scales: List[np.ndarray],
    simcc_split_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    all_keypoints = []
    all_scores = []

    model_input_size_np = np.array(model_input_size, dtype=np.float32)

    for i, output in enumerate(outputs):
        simcc_x, simcc_y = output

        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        keypoints = keypoints / model_input_size_np * scales[i] + centers[i] - scales[i] / 2

        all_keypoints.append(keypoints[0])
        all_scores.append(scores[0])

    return np.array(all_keypoints), np.array(all_scores)


def inference_pose(
    session: ort.InferenceSession,
    out_bbox: Optional[List[List[float]]],
    ori_img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, h, w = session.get_inputs()[0].shape
    model_input_size = (w, h)

    resized_imgs, centers, scales = preprocess(
        ori_img,
        out_bbox=out_bbox,
        input_size=model_input_size,
    )

    outputs = inference(session, resized_imgs)

    keypoints, scores = postprocess(
        outputs,
        model_input_size=model_input_size,
        centers=centers,
        scales=scales,
    )

    return keypoints, scores


def visualize_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_thr: float = 0.3,
) -> np.ndarray:
    vis_img = img.copy()

    for idx, ((x, y), score) in enumerate(zip(keypoints, scores)):
        if score > score_thr:
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                vis_img,
                str(idx),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return vis_img


class RTMPose:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        providers: List[str] = None,
    ):
        if providers is None:
            if device.startswith("cuda"):
                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
            else:
                providers = [
                    "CPUExecutionProvider"
                ]

        self.model_path = model_path
        self.device = device
        self.providers = providers

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        _, _, h, w = self.session.get_inputs()[0].shape
        self.model_input_size = (w, h)

    @classmethod
    def from_config(cls, config_path: str):
        

        cfg = load_yaml(config_path)

        device = get_device(config_path)

        model_path = cfg.get("models", {}) \
                        .get("dwpose", {}) \
                        .get("path", None)

        if model_path is None:
            raise ValueError(
                "Config error: models.dwpose.path not found"
            )

        return cls(
            model_path=model_path,
            device=device,
        )

    def predict(
        self,
        img: np.ndarray,
        out_bbox=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if out_bbox is None:
            out_bbox = []

        keypoints, scores = inference_pose(
            self.session,
            out_bbox,
            img
        )

        return keypoints, scores

    def visualize(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        score_thr: float = 0.3
    ):
        return visualize_keypoints(
            img,
            keypoints,
            scores,
            score_thr
        )

    def predict_and_visualize(
        self,
        img_path: str,
        out_bbox=None,
        score_thr: float = 0.3,
        save_path: str = None,
    ) -> str:
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(
                f"读取图片失败: {img_path}"
            )

        keypoints, scores = self.predict(
            img,
            out_bbox
        )

        vis_img = self.visualize(
            img,
            keypoints[0],
            scores[0],
            score_thr
        )

        if save_path is None:
            base, _ = os.path.splitext(img_path)
            save_path = f"{base}_pose.jpg"

        cv2.imwrite(
            save_path,
            vis_img
        )

        return save_path

if __name__ == "__main__":
    model_path = "./models/dwpose/dw-ll_ucoco_384.onnx"
    img_path = "./assets/3456.png"

    rtmpose = RTMPose(
        model_path=model_path,
        device="cpu",
    )

    save_path = rtmpose.predict_and_visualize(
        img_path=img_path,
        out_bbox=None,
        score_thr=0.3,
    )

    print(f"✅ 可视化结果已保存为: {save_path}")

    vis_img = cv2.imread(save_path)
    cv2.imshow("Keypoints", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()