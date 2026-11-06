"""Shared utilities for VLAC reranking pipeline."""

import os
import tempfile

import cv2
import numpy as np
from PIL import Image


def resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image preserving aspect ratio with zero padding."""
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def center_crop_square(img_pil: Image.Image) -> Image.Image:
    """Center-crop a PIL image to square (shorter side)."""
    w, h = img_pil.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img_pil.crop((left, top, left + side, top + side))


def center_crop_to_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop to target aspect ratio, then resize."""
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h
    if current_aspect > target_aspect:
        # Too wide, crop width
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        img = img[:, offset:offset + new_w]
    else:
        # Too tall, crop height
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        img = img[offset:offset + new_h, :]
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def frames_to_mp4(frames, output_path: str, fps: float = 10.0):
    """Save list of PIL images or numpy arrays as MP4."""
    import imageio
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        writer.append_data(frame)
    writer.close()
    return output_path


def frames_to_temp_mp4(frames, fps: float = 10.0) -> str:
    """Save frames to a temporary MP4 file. Caller must clean up."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    return frames_to_mp4(frames, path, fps=fps)


def split_tracks_78d(tracks_78d: np.ndarray):
    """Split (T, 78) track predictions into view-specific arrays.

    78D = [agentview_mesh(7×2=14) | wrist_grid(25×2=50) | wrist_mesh(7×2=14)]

    Returns:
        agentview_mesh: (T, 7, 2) normalized [0,1]
        wrist_grid: (T, 25, 2) normalized [0,1]
        wrist_mesh: (T, 7, 2) normalized [0,1]
    """
    T = tracks_78d.shape[0]
    flat = tracks_78d.reshape(T, 39, 2)
    agentview_mesh = flat[:, 0:7, :]      # 7 agentview mesh vertices
    wrist_grid = flat[:, 7:32, :]         # 25 CoTracker grid points
    wrist_mesh = flat[:, 32:39, :]        # 7 wrist mesh vertices
    return agentview_mesh, wrist_grid, wrist_mesh
