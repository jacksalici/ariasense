"""M1 checkpoint: render the gaze dot (and a short trail) on the upright RGB stream."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aria_demo.calibration.projection import GazeProjector, Pixel, upright
from aria_demo.sync import History
from aria_demo.types import EyeGazeSample, RgbFrame, Sample

MAX_GAZE_AGE_NS = 60_000_000  # gaze older than 60 ms is not drawn on a frame

DOT_COLOR = (255, 60, 60)  # RGB
TRAIL_COLOR = (255, 200, 0)


@dataclass
class OverlayStats:
    frames: int = 0
    frames_with_gaze: int = 0
    frames_gaze_in_image: int = 0
    blinks: int = 0

    def summary(self) -> str:
        def pct(n: int) -> str:
            return f"{100 * n / self.frames:.0f}%" if self.frames else "-"

        return (
            f"frames={self.frames} · with gaze={self.frames_with_gaze} ({pct(self.frames_with_gaze)}) · "
            f"gaze inside image={self.frames_gaze_in_image} ({pct(self.frames_gaze_in_image)}) · "
            f"blink frames={self.blinks}"
        )


def in_image(px: Pixel, width: int, height: int) -> bool:
    return 0 <= px[0] < width and 0 <= px[1] < height


def draw_gaze(
    image: np.ndarray, gaze_px: Pixel | None, trail: list[Pixel], blinking: bool, label: str
) -> np.ndarray:
    """Draw trail, gaze dot and a status label on an upright RGB image (in place)."""
    h, w = image.shape[:2]
    radius = max(6, w // 80)
    pts = [(int(u), int(v)) for u, v in trail if in_image((u, v), w, h)]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(image, a, b, TRAIL_COLOR, max(2, radius // 3), cv2.LINE_AA)
    if gaze_px is not None and in_image(gaze_px, w, h):
        center = (int(gaze_px[0]), int(gaze_px[1]))
        if blinking:
            cv2.circle(image, center, radius, DOT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, center, radius, DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, center, radius * 3, DOT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(image, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def render(
    samples: Iterable[Sample],
    projector: GazeProjector,
    trail_s: float = 0.5,
    stats: OverlayStats | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (t_ns, upright RGB image with gaze overlay) for every RGB frame."""
    stats = stats if stats is not None else OverlayStats()
    gaze_history: History[EyeGazeSample] = History(window_ns=int(max(trail_s, 0.1) * 2e9))
    t_first: int | None = None

    for sample in samples:
        if isinstance(sample, EyeGazeSample):
            gaze_history.add(sample)
            continue
        if not isinstance(sample, RgbFrame):
            continue

        t_first = sample.t_ns if t_first is None else t_first
        raw_hw = sample.image.shape[:2]
        image = upright(sample.image)
        h, w = image.shape[:2]
        stats.frames += 1

        gaze = gaze_history.latest_before(sample.t_ns, MAX_GAZE_AGE_NS)
        gaze_px = projector.project_sample(gaze, raw_hw) if gaze is not None else None
        blinking = bool(gaze and gaze.left_blink and gaze.right_blink)
        if gaze is not None:
            stats.frames_with_gaze += 1
        if gaze_px is not None and in_image(gaze_px, w, h):
            stats.frames_gaze_in_image += 1
        if blinking:
            stats.blinks += 1

        trail = [
            px
            for g in gaze_history.since(sample.t_ns - int(trail_s * 1e9))
            if g.t_ns <= sample.t_ns and (px := projector.project_sample(g, raw_hw)) is not None
        ]
        label = f"t={(sample.t_ns - t_first) / 1e9:6.2f}s" + ("  BLINK" if blinking else "")
        yield sample.t_ns, draw_gaze(image, gaze_px, trail, blinking, label)


def write_video(frames: Iterable[tuple[int, np.ndarray]], out: Path, show: bool = False) -> int:
    """Write frames to an MP4 (fps estimated from timestamps); optionally show them live."""
    writer: cv2.VideoWriter | None = None
    pending: list[tuple[int, np.ndarray]] = []
    written = 0

    def open_writer(t0: int, t1: int, shape: tuple[int, ...]) -> cv2.VideoWriter:
        fps = 1e9 / (t1 - t0) if t1 > t0 else 30.0
        out.parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shape[1], shape[0]))

    try:
        for t_ns, frame_rgb in frames:
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if show:
                cv2.imshow("aria-demo overlay (q to quit)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer is None:
                pending.append((t_ns, bgr))
                if len(pending) < 2:
                    continue
                writer = open_writer(pending[0][0], pending[1][0], bgr.shape)
                for _, f in pending:
                    writer.write(f)
                    written += 1
                pending.clear()
                continue
            writer.write(bgr)
            written += 1
        if writer is None and pending:  # single-frame recording
            writer = open_writer(0, 0, pending[0][1].shape)
            writer.write(pending[0][1])
            written += 1
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    return written
