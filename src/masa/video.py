#video decoding with explicit source-frame and nominal-time semantics

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float


def inspect_video(path: str | Path) -> VideoInfo:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video. Use a locally decodable MP4, MOV, AVI, or MKV.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width, height, count = (
            int(cap.get(prop))
            for prop in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FRAME_COUNT,
            )
        )
        if not math.isfinite(fps) or fps <= 0 or min(width, height, count) <= 0:
            raise ValueError("Video has invalid frame rate, dimensions, or frame count.")
        ok, _ = cap.read()
        if not ok:
            raise ValueError("Video has no decodable frames.")
        return VideoInfo(str(path), fps, width, height, count, count / fps)
    finally:
        cap.release()


def frame_bounds(
    info: VideoInfo, start_s: float, end_s: float | None, stride: int) -> tuple[int, int]:
    end_s = info.duration_s if end_s is None else end_s
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Frame stride must be a positive integer.")
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        raise ValueError("Clip times must be finite.")
    if start_s < 0 or end_s <= start_s or start_s >= info.duration_s:
        raise ValueError("Select a nonempty interval within the video.")
    if end_s > info.duration_s + 1e-6:
        raise ValueError("End time exceeds the video duration.")
    start = math.ceil(start_s * info.fps - 1e-9)
    stop = min(info.frame_count, math.ceil(end_s * info.fps - 1e-9))
    if start >= stop:
        raise ValueError("Selected interval contains no frames.")
    return start, stop


def read_frame(path: str | Path, frame_index: int) -> np.ndarray:
    info = inspect_video(path)
    if not isinstance(frame_index, int) or not 0 <= frame_index < info.frame_count:
        raise ValueError("Frame index is outside the video.")
    cap = cv2.VideoCapture(info.path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {frame_index}.")
        return frame
    finally:
        cap.release()


def iter_frames(
    path: str | Path, start_s: float = 0.0, end_s: float | None = None, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    info = inspect_video(path)
    start, stop = frame_bounds(info, start_s, end_s, stride)
    cap = cv2.VideoCapture(info.path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, stop):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video decoding stopped unexpectedly at frame {index}.")
            if (index - start) % stride == 0:
                yield index, index / info.fps, frame
    finally:
        cap.release()
