"""Five-window dataloader for ViscNet inference.

A source clip is 60 frames. Five 30-frame windows are cut at frame offsets
{5, 10, 15, 20, 25}; each window is an independent model input, and the five
per-window predictions are averaged into one per-clip estimate (see predict.py).

Preprocessing matches training exactly: decode RGB uint8, (resize to 224 if
needed), then normalize with (x / 127.5) - 1.0  ->  range [-1, 1]. This is a
plain rescale, NOT ImageNet mean/std.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

EVAL_STARTS = (5, 10, 15, 20, 25)  # window start frames (slides by 5)
WINDOW_SIZE = 30                    # frames per window (== model num_frames)
SOURCE_FRAMES = 60                  # frames per source clip
IMAGE_SIZE = 224


def read_source_frames(path: str, min_frames: int = SOURCE_FRAMES) -> list[np.ndarray]:
    """Decode an mp4 into a list of [H, W, 3] uint8 RGB frames."""
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) < min_frames:
        raise ValueError(f"{path}: decoded {len(frames)} frames < required {min_frames}")
    return frames


def window_to_uint8(frames, start, window_size=WINDOW_SIZE, image_size=IMAGE_SIZE):
    """Cut one [window_size, H, W, 3] uint8 window and resize to 224 if needed."""
    clip = frames[start : start + window_size]
    if len(clip) < window_size:
        raise ValueError(f"window start {start} needs {window_size} frames, got {len(clip)}")
    out = []
    for f in clip:
        if f.shape[0] != image_size or f.shape[1] != image_size:
            f = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        out.append(f)
    return torch.from_numpy(np.stack(out, axis=0))  # uint8 [T, H, W, 3]


def normalize_clip_batch(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [N, T, H, W, 3] -> float [N, T, 3, H, W] in [-1, 1]."""
    return frames.to(torch.float32).div_(127.5).sub_(1.0).permute(0, 1, 4, 2, 3).contiguous()


def five_window_batch(path: str) -> torch.Tensor:
    """Decode a clip and stack its 5 eval windows into a uint8 [5, T, H, W, 3] batch."""
    frames = read_source_frames(path)
    windows = [window_to_uint8(frames, s) for s in EVAL_STARTS]
    return torch.stack(windows, dim=0)
