#ultralytics boundary: preserve raw confidence and never fabricate fighter identities

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
OFFICIAL_MODELS = ("yolo11n-pose.pt", "yolo11s-pose.pt")


def convert_result(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.keypoints is None or result.keypoints.data.shape[1:] != (17, 3):
        raise ValueError("This application requires a 17-keypoint COCO pose model with confidence.")
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    points = result.keypoints.data.cpu().numpy()
    people = []
    for index, (box, score, cls, joints) in enumerate(
        zip(boxes, scores, classes, points, strict=True)
    ):
        if int(cls) != 0 or not np.isfinite(box).all() or not math.isfinite(float(score)):
            continue
        keypoints = []
        for name, (x, y, confidence) in zip(COCO_KEYPOINT_NAMES, joints, strict=True):
            valid = all(math.isfinite(float(v)) for v in (x, y, confidence)) and confidence > 0
            keypoints.append(
                {
                    "name": name,
                    "x": float(x) if valid else None,
                    "y": float(y) if valid else None,
                    "confidence": float(confidence) if valid else 0.0,
                }
            )
        track_id = int(ids[index]) if ids is not None and math.isfinite(float(ids[index])) else None
        people.append(
            {
                "track_id": track_id,
                "bbox_xyxy": [float(v) for v in box],
                "confidence": float(score),
                "keypoints": keypoints,
            }
        )
    return people


class YOLOPoseEstimator:
    def __init__(self, config):
        if config.model not in OFFICIAL_MODELS:
            raise ValueError(
                "Use an approved pretrained YOLO11 pose model; arbitrary weights are disabled."
            )
        #all tracker dependencies are declared never let inference pip-install packages.
        from masa.privacy import configure_ultralytics_privacy

        configure_ultralytics_privacy()
        import torch
        import ultralytics
        from ultralytics import YOLO

        if config.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable. Select CPU on this machine.")
        if config.device == "cuda:0" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Select CPU on this machine.")
        weights = Path("weights") / config.model
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights), task="pose")
        self.config = config
        self.metadata = {
            "name": config.model,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "keypoint_format": "COCO-17",
            "imgsz": 640,
            "device": config.device,
            "usage_telemetry": False,
        }

    def detect(self, frame: np.ndarray) -> list[dict]:
        maximum = self.config.expected_people if self.config.selection_mode == "automatic" else 300
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker,
            conf=self.config.confidence,
            max_det=maximum,
            device=self.config.device,
            imgsz=640,
            classes=[0],
            save=False,
            verbose=False,
        )
        return convert_result(results[0])
