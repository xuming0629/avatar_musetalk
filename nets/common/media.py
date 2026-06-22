from pathlib import Path
import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_video_frames(video_path: str | Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def read_image_as_frame(path: str | Path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def write_video(frames, out_path: str | Path, fps: int = 25):
    if not frames:
        raise ValueError("empty frames")
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return out_path


def make_mock_video(out_path: str | Path, text: str = "XumingAvatar Mock", fps: int = 25, seconds: int = 3):
    frames = []
    w, h = 720, 1280
    for i in range(fps * seconds):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f"frame: {i}", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cx = w // 2
        cy = h // 2
        cv2.circle(img, (cx, cy - 100), 140, (180, 180, 180), -1)
        mouth_w = 90 + int(40 * abs(np.sin(i / 4)))
        cv2.ellipse(img, (cx, cy - 40), (mouth_w, 25), 0, 0, 360, (30, 30, 30), -1)
        frames.append(img)
    return write_video(frames, out_path, fps)
