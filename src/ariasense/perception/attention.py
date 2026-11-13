"""Which detected object is the wearer looking at, and for how long."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ariasense.calibration.projection import Pixel
from ariasense.types import Detection


def box_distance(px: Pixel, box: tuple[float, float, float, float]) -> float:
    """Distance from a point to a box (0 if inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px[0], 0.0, px[0] - x2)
    dy = max(y1 - px[1], 0.0, px[1] - y2)
    return (dx * dx + dy * dy) ** 0.5


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_attended(
    detections: Sequence[Detection], gaze_px: Pixel | None, max_dist_px: float
) -> Detection | None:
    """The detection under the gaze point.

    If several boxes contain the gaze (e.g. a cup on a dining table), the smallest
    one wins, as it is the most specific. Otherwise the nearest box within
    `max_dist_px` is taken, to absorb small gaze/calibration errors.
    """
    if gaze_px is None or not detections:
        return None
    containing = [d for d in detections if box_distance(gaze_px, d.box) == 0.0]
    if containing:
        return min(containing, key=lambda d: d.area)
    nearest = min(detections, key=lambda d: box_distance(gaze_px, d.box))
    return nearest if box_distance(gaze_px, nearest.box) <= max_dist_px else None


@dataclass
class AttentionState:
    target: Detection | None
    dwell_s: float
    totals_s: dict[str, float] = field(default_factory=dict)
    switches: int = 0


class AttentionTracker:
    """Tracks the attended object over time: dwell on the current one, totals per class.

    Short gaps (blinks, saccades, a missed detection) shorter than `grace_s` do not
    reset the dwell.
    """

    def __init__(self, grace_s: float = 0.3, iou_match: float = 0.3) -> None:
        self.grace_ns = int(grace_s * 1e9)
        self.iou_match = iou_match
        self._current: Detection | None = None
        self._dwell_ns = 0
        self._last_seen_t: int | None = None
        self._totals_ns: dict[str, int] = {}
        self._switches = 0

    def _same(self, a: Detection, b: Detection) -> bool:
        if a.track_id is not None and b.track_id is not None:
            return a.track_id == b.track_id
        return a.class_name == b.class_name and iou(a.box, b.box) >= self.iou_match

    def update(self, t_ns: int, target: Detection | None) -> AttentionState:
        if target is not None and self._current is not None and self._same(target, self._current):
            # Time since last seen, so a short gap (e.g. a blink) still counts as dwell.
            dt = 0 if self._last_seen_t is None else max(0, t_ns - self._last_seen_t)
            self._dwell_ns += dt
            self._totals_ns[target.class_name] = self._totals_ns.get(target.class_name, 0) + dt
            self._current, self._last_seen_t = target, t_ns
        elif target is not None:
            self._current, self._dwell_ns, self._last_seen_t = target, 0, t_ns
            self._switches += 1
        elif (
            self._current is not None
            and self._last_seen_t is not None
            and t_ns - self._last_seen_t > self.grace_ns
        ):
            self._current, self._dwell_ns = None, 0

        return AttentionState(
            target=self._current,
            dwell_s=self._dwell_ns / 1e9,
            totals_s={k: v / 1e9 for k, v in self._totals_ns.items()},
            switches=self._switches,
        )
