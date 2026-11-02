#experimental image-plane movement no strikes, intent or tactical success

import math
from collections import Counter

import numpy as np


class MovementAnalysis:
    def __init__(self, width, height, threshold, *, fixed_camera=False,
                 smoothing_s=0.08, max_gap_s=0.5, min_phase_s=0.15):
        self.width, self.height, self.threshold = width, height, threshold
        self.fixed_camera = fixed_camera
        self.smoothing_s, self.max_gap_s, self.min_phase_s = smoothing_s, max_gap_s, min_phase_s
        self.enter = math.hypot(width, height) * 0.01
        self.release = self.enter / 2
        self.previous = None
        self.paths = {role: [] for role in "AB"}
        self.directions = {role: "unclear" for role in "AB"}
        self.phase, self.pending, self.pending_since = "uncertain", None, None
        self.rows = []
        self.tracks = None

    def _hip(self, person):
        coordinates = []
        for name in ("left_hip", "right_hip"):
            point = next((p for p in person["keypoints"] if p["name"] == name), None)
            if point is None or any(not isinstance(point.get(key), (int, float))
                                    or not math.isfinite(point[key])
                                    for key in ("x", "y", "confidence")):
                return None
            if (point["confidence"] < self.threshold
                    or not 0 <= point["x"] < self.width or not 0 <= point["y"] < self.height):
                return None
            coordinates.append([point["x"], point["y"]])
        return np.mean(coordinates, axis=0)

    def _reset(self):
        self.previous = None
        self.paths = {role: [] for role in "AB"}
        self.directions = {role: "unclear" for role in "AB"}
        self.phase, self.pending, self.pending_since = "uncertain", None, None

    def _direction(self, role, value):
        previous = self.directions[role]
        if value >= self.enter or (previous == "toward" and value >= self.release):
            return "toward"
        if value <= -self.enter or (previous == "away" and value <= -self.release):
            return "away"
        return "unclear"

    def observe(self, record):
        time = record["timestamp_s"]
        if not math.isfinite(time) or (self.rows and time <= self.rows[-1]["timestamp_s"]):
            raise ValueError("Movement requires finite, increasing source timestamps.")
        row = {"frame_index": record["frame_index"], "timestamp_s": time,
               "centers": {role: None for role in "AB"},
               "velocity_px_s": {role: None for role in "AB"},
               "approach_px_s": {role: None for role in "AB"},
               "direction": {role: "unclear" for role in "AB"},
               "distance_px": None, "distance_rate_px_s": None,
               "distance_delta_px": None, "interval_s": None,
               "paths": {role: [] for role in "AB"},
               "phase": "uncertain", "status": "excluded", "reasons": []}
        centers = {}
        people = {role: [p for p in record["people"] if p.get("fighter") == role] for role in "AB"}
        if any(len(people[role]) != 1 for role in "AB"):
            row["reasons"].append("selected_fighter_missing_or_ambiguous")
        else:
            people = {role: people[role][0] for role in "AB"}
            identifiers = {role: people[role]["track_id"] for role in "AB"}
            if self.tracks is None:
                self.tracks = identifiers
            if (identifiers != self.tracks or None in identifiers.values()
                    or len(set(identifiers.values())) != 2):
                row["reasons"].append("fighter_identity_uncertain")
            raw = {role: self._hip(people[role]) for role in "AB"}
            if any(center is None for center in raw.values()):
                row["reasons"].append("reliable_hips_missing")
            centers = {role: center for role, center in raw.items() if center is not None}
        row["reasons"].extend(flag for flag in record.get("flags", [])
                              if "identity" in flag or "missing" in flag or "unassigned" in flag)
        if row["reasons"]:
            self._reset()
        else:
            previous = self.previous
            dt = time - previous[0] if previous else None
            if dt is not None and dt > self.max_gap_s:
                self._reset()
                row["reasons"].append("long_gap_velocity_reset")
                dt = None
                previous = None
            if dt is not None and previous is not None and self.smoothing_s > 0:
                alpha = 1 - math.exp(-dt / self.smoothing_s)
                centers = {role: previous[1][role] + alpha * (
                    centers[role] - previous[1][role]) for role in "AB"}
            distance = float(np.linalg.norm(centers["B"] - centers["A"]))
            if distance < 1e-6:
                row["reasons"].append("coincident_centers")
                self._reset()
                self.rows.append(row)
                return row
            row["centers"] = {role: centers[role].tolist() for role in "AB"}
            row["distance_px"] = distance
            for role in "AB":
                self.paths[role].append((time, centers[role].tolist()))
                self.paths[role] = [(t, p) for t, p in self.paths[role] if time - t <= 2][-120:]
                row["paths"][role] = [p for _, p in self.paths[role]]
            row["status"] = "uncertain"
            if dt is None or previous is None:
                row["reasons"].append("velocity_warmup")
            else:
                velocities = {role: (centers[role] - previous[1][role]) / dt for role in "AB"}
                row["velocity_px_s"] = {role: velocities[role].tolist() for role in "AB"}
                row["distance_rate_px_s"] = (distance - previous[2]) / dt
                row["distance_delta_px"], row["interval_s"] = distance - previous[2], dt
                if not self.fixed_camera:
                    row["reasons"].append("camera_not_confirmed_fixed")
                else:
                    axis = (centers["B"] - centers["A"]) / distance
                    values = {"A": float(velocities["A"] @ axis),
                              "B": -float(velocities["B"] @ axis)}
                    row["approach_px_s"] = values
                    self.directions = {role: self._direction(role, values[role]) for role in "AB"}
                    row["direction"] = dict(self.directions)
                    toward = [role for role in "AB" if self.directions[role] == "toward"]
                    if row["distance_rate_px_s"] >= self.enter:
                        candidate = "gap_opens"
                    elif len(toward) == 2:
                        candidate = "both_close"
                    elif toward:
                        candidate = f"{toward[0].lower()}_closes"
                    else:
                        candidate = "uncertain"
                    if candidate != self.pending:
                        self.pending, self.pending_since = candidate, time
                    assert self.pending_since is not None
                    if time - self.pending_since >= self.min_phase_s:
                        self.phase = candidate
                    row["phase"] = self.phase
                    row["pending_phase"] = candidate
                    row["status"] = "reliable" if self.phase != "uncertain" else "uncertain"
                    if row["status"] == "uncertain":
                        row["reasons"].append("below_threshold_or_phase_pending")
            self.previous = (time, centers, distance)
        self.rows.append(row)
        return row

    def report(self, end_frame, fps):
        phases = []
        for index, row in enumerate(self.rows):
            boundary = (self.rows[index + 1]["frame_index"]
                        if index + 1 < len(self.rows) else end_frame)
            if not phases or (row["phase"], row["status"]) != (
                    phases[-1]["label"], phases[-1]["status"]):
                phases.append({"id": len(phases), "label": row["phase"], "status": row["status"],
                               "start_frame": row["frame_index"], "start_s": row["timestamp_s"],
                               "first_ordinal": index, "review": "not_reviewed", "reasons": []})
            phases[-1]["end_frame_exclusive"] = boundary
            phases[-1]["end_s"] = boundary / fps
            phases[-1]["reasons"] = sorted(set(phases[-1]["reasons"]) | set(row["reasons"]))
        return {"enabled": True, "experimental": True, "fixed_camera_confirmed": self.fixed_camera,
                "human_reviewed_intervals": 0,
                "parameters": {"smoothing_s": self.smoothing_s, "max_gap_s": self.max_gap_s,
                               "min_phase_s": self.min_phase_s, "enter_px_s": self.enter,
                               "release_px_s": self.release, "path_tail_s": 2},
                "frame_counts": dict(Counter(row["status"] for row in self.rows)),
                "interval_counts": dict(Counter(phase["status"] for phase in phases)),
                "phases": phases,
                "limitations": ["Uncalibrated 2D geometry, not strikes, intent or success.",
                                "Camera motion and perspective can look like fighter motion.",
                                "A/B tracker swaps may go unflagged; human review is required."]}
