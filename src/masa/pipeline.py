#reproducible video analysis
#a completed manifest is written only after all artifacts succeed

from __future__ import annotations
from masa.features import MotionFeatures
from masa.fighter_selection import FighterSelection, validate_selection
from masa.movement import MovementAnalysis
from masa.tracking import TrackQuality


import csv
import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import imageio_ffmpeg
import numpy as np

from masa.video import frame_bounds, inspect_video, iter_frames

FEATURE_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "track_id",
    "left_elbow_deg",
    "right_elbow_deg",
    "left_knee_deg",
    "right_knee_deg",
    "left_wrist_speed_body_per_s",
    "right_wrist_speed_body_per_s",
    "relative_distance_body",
]


class Estimator(Protocol):
    metadata: dict
    def detect(self, frame: np.ndarray) -> list[dict]: ...


@dataclass(frozen=True)
class RunConfig:
    model: str = "yolo11n-pose.pt"
    tracker: str = "botsort.yaml"
    device: str = "cpu"
    start_s: float = 0.0
    end_s: float | None = None
    stride: int = 1
    confidence: float = 0.25
    keypoint_confidence: float = 0.35
    expected_people: int = 2
    rights_note: str = ""
    source_note: str = ""
    selection_mode: str = "automatic"
    fighter_boxes: list[list[float]] | None = None
    fighter_seed_frame: int | None = None
    fixed_camera: bool = False

    def validate(self) -> None:
        for value in (self.confidence, self.keypoint_confidence):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Confidence thresholds must be within (0, 1].")
        if self.expected_people not in (1, 2):
            raise ValueError("Expected people must be 1 or 2.")
        if type(self.fixed_camera) is not bool:
            raise ValueError("Fixed camera confirmation must be a boolean.")
        if self.tracker not in ("botsort.yaml", "bytetrack.yaml"):
            raise ValueError("Choose botsort.yaml or bytetrack.yaml.")
        if self.device not in ("cpu", "mps", "cuda:0"):
            raise ValueError("Device must be cpu, mps, or cuda:0.")
        if not self.rights_note.strip():
            raise ValueError("Record footage rights or permission notes before analysis.")
        validate_selection(self.selection_mode, self.expected_people,
                           self.fighter_boxes, self.fighter_seed_frame)


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    manifest_path: Path
    poses_path: Path
    video_path: Path
    features_path: Path
    contact_sheet_path: Path
    frame_count: int


def draw_overlay(frame: np.ndarray, record: dict, threshold: float) -> np.ndarray:
    from masa.pose import COCO_EDGES

    image = frame.copy()
    palette = [(255, 180, 70), (90, 220, 130), (230, 130, 230), (70, 200, 255)]
    height, width = image.shape[:2]
    for person in record["people"]:
        track_id = person["track_id"]
        color = palette[track_id % len(palette)] if track_id is not None else (160, 160, 160)
        fighter = person.get("fighter")
        if fighter in ("A", "B"):
            color = palette[("A", "B").index(fighter)]
        x1, y1, x2, y2 = [round(v) for v in person["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        label = f"Track {track_id}" if track_id is not None else "Unknown track"
        if fighter in ("A", "B"):
            label = f"Fighter {fighter} / {label}"
        cv2.putText(
            image,
            f"{label} {person['confidence']:.2f}",
            (max(2, x1), max(43, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
        joints = person["keypoints"]
        visible = [
            p["x"] is not None
            and p["y"] is not None
            and 0 <= p["x"] < width
            and 0 <= p["y"] < height
            for p in joints
        ]
        for a, b in COCO_EDGES:
            if (
                visible[a]
                and visible[b]
                and min(joints[a]["confidence"], joints[b]["confidence"]) >= threshold
            ):
                cv2.line(
                    image,
                    (round(joints[a]["x"]), round(joints[a]["y"])),
                    (round(joints[b]["x"]), round(joints[b]["y"])),
                    color,
                    2,
                )
        for point, is_visible in zip(joints, visible, strict=True):
            if is_visible:
                reliable = point["confidence"] >= threshold
                cv2.circle(
                    image,
                    (round(point["x"]), round(point["y"])),
                    3,
                    color if reliable else (150, 150, 150),
                    -1 if reliable else 1,
                )
    movement = record.get("movement", {})
    for role, center in movement.get("centers", {}).items():
        if center is None:
            continue
        color = palette[("A", "B").index(role)]
        path = movement["paths"][role]
        if len(path) > 1:
            cv2.polylines(image, [np.array(path, dtype=np.int32)], False, color, 2)
        cv2.circle(image, tuple(round(value) for value in center), 6, color, 2)
    text = f"Frame {record['frame_index']} | {record['timestamp_s']:.3f}s"
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (18, 22, 30), -1)
    cv2.putText(image, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if record["flags"]:
        cv2.putText(
            image,
            ", ".join(record["flags"]),
            (6, image.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 190, 255),
            1,
        )
    return image


def analyze_video(
    source: str | Path,
    output_dir: str | Path,
    config: RunConfig,
    progress: Callable[[float, str], None] | None = None,
    *,
    estimator: Estimator | None = None,
) -> RunResult:
    config.validate()
    info = inspect_video(source)
    start, stop = frame_bounds(info, config.start_s, config.end_s, config.stride)
    validate_selection(config.selection_mode, config.expected_people, config.fighter_boxes,
                       config.fighter_seed_frame, info=info, first_frame=start)
    total = len(range(start, stop, config.stride))
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; choose a new run directory.")
    if estimator is None:
        from masa.pose import YOLOPoseEstimator

        estimator = YOLOPoseEstimator(config)
    digest = hashlib.sha256()
    with Path(info.path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = RunResult(
        output_dir,
        output_dir / "manifest.json",
        output_dir / "poses.jsonl",
        output_dir / "overlay.mp4",
        output_dir / "features.csv",
        output_dir / "contact_sheet.jpg",
        total,
    )
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(
            str(result.video_path),
            (info.width, info.height),
            pix_fmt_in="bgr24",
            fps=info.fps / config.stride,
            codec="libx264",
            macro_block_size=1,
            output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart"],
        )
        writer.send(None)
        thumbnails = []
        thumb_indices = set(np.linspace(0, total - 1, min(total, 12), dtype=int).tolist())
        summary = {
            "processed_frames": 0,
            "detections": 0,
            "frames_with_missing_people": 0,
            "track_ids": [],
        }
        

        motion = MotionFeatures(config.keypoint_confidence)
        track_quality = TrackQuality(config.expected_people)
        selection = FighterSelection(config.expected_people, boxes=config.fighter_boxes,
                                     seed_frame=config.fighter_seed_frame)
        movement = (MovementAnalysis(info.width, info.height, config.keypoint_confidence,
                                    fixed_camera=config.fixed_camera)
                    if config.selection_mode == "manual" and config.expected_people == 2 else None)
        track_ids: set[int] = set()
        with (
            result.poses_path.open("w") as poses,
            result.features_path.open("w", newline="") as features,
        ):
            feature_writer = csv.DictWriter(features, fieldnames=FEATURE_COLUMNS)
            feature_writer.writeheader()
            for ordinal, (index, timestamp, frame) in enumerate(
                iter_frames(source, config.start_s, config.end_s, config.stride)
            ):
                people, selection_flags = selection.select(estimator.detect(frame))
                flags = track_quality.observe(people) + selection_flags
                if "fewer_people_than_expected" in flags:
                    summary["frames_with_missing_people"] += 1
                record = {
                    "frame_index": index,
                    "timestamp_s": timestamp,
                    "people": people,
                    "flags": flags,
                }
                if movement is not None:
                    record["movement"] = movement.observe(record)
                poses.write(json.dumps(record, allow_nan=False) + "\n")
                feature_writer.writerows(motion.observe(record))
                overlay = draw_overlay(frame, record, config.keypoint_confidence)
                writer.send(np.ascontiguousarray(overlay))
                if ordinal in thumb_indices:
                    thumbnails.append(cv2.resize(overlay, (320, 240)))
                track_ids.update(p["track_id"] for p in people if p["track_id"] is not None)
                summary["processed_frames"] += 1
                summary["detections"] += len(people)
                if progress:
                    progress((ordinal + 1) / total, f"Processed source frame {index}")
        writer.close()
        writer = None
        while len(thumbnails) % 3:
            thumbnails.append(np.zeros_like(thumbnails[0]))
        sheet = np.vstack([np.hstack(thumbnails[i : i + 3]) for i in range(0, len(thumbnails), 3)])
        if not cv2.imwrite(str(result.contact_sheet_path), sheet):
            raise OSError("Cannot write contact sheet.")
        summary["track_ids"] = sorted(track_ids)
        from masa.pose import COCO_KEYPOINT_NAMES

        manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "coordinates": "source_image_pixels_xy_top_left",
            "video": asdict(info),
            "source_sha256": digest.hexdigest(),
            "config": asdict(config),
            "model": estimator.metadata,
            "selection": selection.metadata,
            "movement": movement.report(stop, info.fps) if movement is not None else {
                "enabled": False, "reason": "Select two fighters manually for movement analysis."},
            "summary": summary,
            "outputs": {
                "poses": "poses.jsonl",
                "overlay": "overlay.mp4",
                "features": "features.csv",
                "contact_sheet": "contact_sheet.jpg",
            },
            "limitations": [
                "Track IDs are not verified identities. Switches may be unflagged.",
                "Time = source frame index / nominal FPS; variable-rate PTS is not supported.",
                "Overlay contains sampled frames without audio; its FPS is source FPS / stride.",
                "2D pose features are view-dependent, not physical distances or action labels.",
            ],
        }
        result.manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return result
    except BaseException as error:
        try:
            if writer is not None:
                try:
                    writer.close()
                except Exception as close_error:
                    error.add_note(f"Writer close also failed: {close_error}")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
