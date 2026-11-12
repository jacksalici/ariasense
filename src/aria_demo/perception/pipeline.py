"""Frames with gaze -> YOLO detections -> attended object and dwell."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from aria_demo.overlay import GazeFrame, draw_detections, draw_gaze, draw_status
from aria_demo.perception.attention import AttentionState, AttentionTracker, pick_attended
from aria_demo.types import Detection

Detector = Callable[[np.ndarray], list[Detection]]


@dataclass
class PerceptionFrame:
    frame: GazeFrame
    detections: list[Detection]
    attention: AttentionState


def is_wearer_body(d: Detection, image_h: int, bottom_frac: float = 0.95) -> bool:
    """Heuristic: a 'person' box reaching the bottom edge is the wearer's own arm/hand.

    Seen on s01_desk: YOLO labels the wearer's forearm 'person', which then steals
    attention from the object being handled. Replace with Aria hand tracking later.
    """
    return d.class_name == "person" and d.box[3] >= bottom_frac * image_h


def perceive(
    frames: Iterable[GazeFrame],
    detector: Detector,
    tracker: AttentionTracker | None = None,
    snap_frac: float = 0.03,
    ignore_wearer_body: bool = True,
) -> Iterator[PerceptionFrame]:
    """Run the detector on each frame and work out the attended object.

    `snap_frac` is how far (as a fraction of image width) the gaze may fall outside a
    box and still count as looking at it.
    """
    tracker = tracker or AttentionTracker()
    for frame in frames:
        detections = detector(frame.image)
        if ignore_wearer_body:
            h = frame.image.shape[0]
            detections = [d for d in detections if not is_wearer_body(d, h)]
        max_dist = snap_frac * frame.image.shape[1]
        target = pick_attended(detections, frame.gaze_px, max_dist) if not frame.blinking else None
        yield PerceptionFrame(frame, detections, tracker.update(frame.t_ns, target))


def draw(pf: PerceptionFrame) -> np.ndarray:
    image = pf.frame.image
    att = pf.attention
    draw_detections(image, pf.detections, att.target if att.target in pf.detections else None, att.dwell_s)
    draw_gaze(image, pf.frame)
    looking = f"looking at: {att.target.class_name}" if att.target else "looking at: -"
    return draw_status(image, pf.frame, looking)
