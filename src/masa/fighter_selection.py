#retain at most the requested number; this is not athlete recognition

import math
from itertools import product


def validate_selection(mode, maximum, boxes, seed_frame, *, info=None, first_frame=None):
    if mode not in ("automatic", "manual"):
        raise ValueError("Choose automatic or manual fighter selection.")
    if mode == "automatic":
        if boxes is not None or seed_frame is not None:
            raise ValueError("Fighter boxes require manual selection mode.")
        return
    if not isinstance(boxes, (list, tuple)) or len(boxes) != maximum:
        raise ValueError("Draw one fighter box per selected person; two boxes for two fighters.")
    if type(seed_frame) is not int or seed_frame < 0:
        raise ValueError("The fighter selection must record a nonnegative source frame index.")
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Fighter boxes need finite, ordered source-pixel coordinates.")
        if info is not None and (box[2] > info.width or box[3] > info.height):
            raise ValueError("Fighter boxes must stay within the source dimensions.")
    if len({tuple(box) for box in boxes}) != maximum:
        raise ValueError("Draw distinct fighter boxes for A and B.")
    if first_frame is not None and seed_frame != first_frame:
        raise ValueError("Draw the fighter boxes on the first analyzed source frame.")


def _iou(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0


class FighterSelection:
    def __init__(self, maximum, *, boxes=None, seed_frame=None):
        self.maximum = maximum
        self.boxes = tuple(tuple(box) for box in boxes) if boxes is not None else None
        self.seed_frame = seed_frame
        self.tracks = {}

    @property
    def metadata(self):
        return {
            "method": "manual_boxes" if self.boxes is not None else "confidence_cap",
            "maximum_people": self.maximum,
            "seed_frame_index": self.seed_frame,
            "boxes_xyxy": self.boxes,
            "fighter_tracks": dict(self.tracks),
            "initialization_min_iou": 0.25 if self.boxes is not None else None,
            "initialization_ambiguity_margin": 0.05 if self.boxes is not None else None,
        }

    def _bind(self, people):
        assert self.boxes is not None
        matches = [[(person, _iou(box, person["bbox_xyxy"])) for person in people
                    if person["track_id"] is not None
                    and _iou(box, person["bbox_xyxy"]) >= 0.25] for box in self.boxes]
        assignments = [
            (sum(score for _, score in choices), choices) for choices in product(*matches)
            if len({person["track_id"] for person, _ in choices}) == self.maximum
        ]
        assignments.sort(key=lambda item: item[0], reverse=True)
        if not assignments or (len(assignments) > 1
                               and assignments[0][0] - assignments[1][0] <= 0.05):
            raise ValueError(
                "Could not match the initial fighter selection to distinct tracked people. "
                "Redraw the boxes or choose a clearer clip-start frame; no fallback was used."
            )
        self.tracks = {label: person["track_id"] for label, (person, _) in
                       zip("AB", assignments[0][1], strict=False)}

    def select(self, people):
        if self.boxes is None:
            return sorted(people, key=lambda person: person["confidence"],
                          reverse=True)[:self.maximum], []
        if not self.tracks:
            self._bind(people)
        selected, flags = [], []
        for label, identifier in self.tracks.items():
            matches = [person for person in people if person["track_id"] == identifier]
            if len(matches) == 1:
                selected.append({**matches[0], "fighter": label})
            else:
                flags.append(f"fighter_{label.lower()}_missing")
                if len(matches) > 1:
                    flags.append("fighter_identity_uncertain")
        return selected, flags
