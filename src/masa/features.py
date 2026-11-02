#view dependent 2D measurements
#missing evidence stays missing, never zero-filled

from __future__ import annotations

import numpy as np


def reliable_point(person: dict, name: str, threshold: float) -> np.ndarray | None:
    point = next((p for p in person["keypoints"] if p["name"] == name), None)
    if point is None or point["confidence"] < threshold or point["x"] is None or point["y"] is None:
        return None
    result = np.array([point["x"], point["y"]], dtype=float)
    return result if np.isfinite(result).all() else None


def angle(a, b, c) -> float | None:
    if a is None or b is None or c is None:
        return None
    left, right = a - b, c - b
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right) / denominator, -1, 1))))


def torso(person, threshold) -> tuple[np.ndarray | None, float | None]:
    points = [
        reliable_point(person, name, threshold)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    ]
    if any(point is None for point in points):
        return None, None
    shoulder = (points[0] + points[1]) / 2
    hip = (points[2] + points[3]) / 2
    size = float(np.linalg.norm(shoulder - hip))
    return (hip, size) if size > 1e-8 else (None, None)


class MotionFeatures:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.previous: dict = {}

    def observe(self, record: dict) -> list[dict]:
        timestamp = record["timestamp_s"]
        people = record["people"]
        identity_uncertain = any("identity_uncertain" in flag for flag in record.get("flags", []))
        bodies = [torso(person, self.threshold) for person in people]
        distance = None
        if len(bodies) == 2 and all(size is not None for _, size in bodies):
            distance = float(
                np.linalg.norm(bodies[0][0] - bodies[1][0]) / ((bodies[0][1] + bodies[1][1]) / 2)
            )
        rows = []
        current = {}
        for person, (_, scale) in zip(people, bodies, strict=True):
            track_id = person["track_id"]
            row = {
                "frame_index": record["frame_index"],
                "timestamp_s": timestamp,
                "track_id": track_id,
                "relative_distance_body": distance,
            }
            for side in ("left", "right"):
                for joint, names in (
                    ("elbow", ("shoulder", "elbow", "wrist")),
                    ("knee", ("hip", "knee", "ankle")),
                ):
                    row[f"{side}_{joint}_deg"] = angle(
                        *(
                            reliable_point(person, f"{side}_{name}", self.threshold)
                            for name in names
                        )
                    )
                wrist = reliable_point(person, f"{side}_wrist", self.threshold)
                key = (track_id, side)
                speed = None
                previous = self.previous.get(key)
                if (
                    not identity_uncertain
                    and track_id is not None
                    and wrist is not None
                    and scale is not None
                ):
                    if previous is not None and timestamp > previous[0]:
                        speed = float(
                            np.linalg.norm(wrist - previous[1])
                            / ((timestamp - previous[0]) * scale)
                        )
                    current[key] = (timestamp, wrist)
                row[f"{side}_wrist_speed_body_per_s"] = speed
            rows.append(row)
        self.previous = current
        return rows
