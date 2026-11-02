#conservative review flags, not a claim to detect every identity switch

from itertools import combinations


class TrackQuality:
    def __init__(self, expected_people: int):
        self.expected_people = expected_people
        self.previous_ids: set[int] = set()
        self.seen_ids: set[int] = set()

    def observe(self, people: list[dict]) -> list[str]:
        flags = []
        ids = {p["track_id"] for p in people if p["track_id"] is not None}
        if len(people) < self.expected_people:
            flags.append("fewer_people_than_expected")
        if len(people) > self.expected_people:
            flags.append("extra_people_visible")
        if any(p["track_id"] is None for p in people):
            flags.append("unassigned_track")
        if ids & (self.seen_ids - self.previous_ids):
            flags.append("reappeared_track_identity_uncertain")
        for first, second in combinations(people, 2):
            a, b = first["bbox_xyxy"], second["bbox_xyxy"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (
                max(0, a[2] - a[0]) * max(0, a[3] - a[1])
                + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                - intersection
            )
            if union > 0 and intersection / union >= 0.15:
                flags.append("overlapping_tracks_identity_uncertain")
                break
        self.previous_ids = ids
        self.seen_ids |= ids
        return flags
