"""Align gaze with RGB frames and draw overlays (gaze dot, trail, detections, attention)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aria_demo.calibration.projection import GazeProjector, Pixel, upright
from aria_demo.sync import History
from aria_demo.types import Detection, EyeGazeSample, RgbFrame, Sample

MAX_GAZE_AGE_NS = 60_000_000  # gaze older than 60 ms is not drawn on a frame

# Colours are RGB.
DOT_COLOR = (255, 60, 60)
TRAIL_COLOR = (255, 200, 0)
BOX_COLOR = (160, 160, 160)
ATTENDED_COLOR = (60, 220, 90)
WHITE = (255, 255, 255)


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


@dataclass
class GazeFrame:
    """One RGB frame (display orientation) with the gaze aligned to it."""

    t_ns: int
    t_rel_s: float
    image: np.ndarray
    gaze: EyeGazeSample | None
    gaze_px: Pixel | None  # None if no gaze, invalid gaze, or outside the image
    trail: list[Pixel]
    blinking: bool


def in_image(px: Pixel, width: int, height: int) -> bool:
    return 0 <= px[0] < width and 0 <= px[1] < height


def gaze_frames(
    samples: Iterable[Sample],
    projector: GazeProjector,
    trail_s: float = 0.5,
    stats: OverlayStats | None = None,
) -> Iterator[GazeFrame]:
    """Yield a `GazeFrame` for every RGB frame, using the latest gaze before it."""
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
        image = upright(sample.image) if projector.rotate_upright else sample.image.copy()
        h, w = image.shape[:2]
        stats.frames += 1

        gaze = gaze_history.latest_before(sample.t_ns, MAX_GAZE_AGE_NS)
        gaze_px = projector.project_sample(gaze, raw_hw) if gaze is not None else None
        if gaze_px is not None and not in_image(gaze_px, w, h):
            gaze_px = None
        blinking = bool(gaze and gaze.left_blink and gaze.right_blink)
        stats.frames_with_gaze += gaze is not None
        stats.frames_gaze_in_image += gaze_px is not None
        stats.blinks += blinking

        trail = [
            px
            for g in gaze_history.since(sample.t_ns - int(trail_s * 1e9))
            if g.t_ns <= sample.t_ns and (px := projector.project_sample(g, raw_hw)) is not None
        ]
        yield GazeFrame(
            t_ns=sample.t_ns,
            t_rel_s=(sample.t_ns - t_first) / 1e9,
            image=image,
            gaze=gaze,
            gaze_px=gaze_px,
            trail=trail,
            blinking=blinking,
        )


def _scale(image: np.ndarray) -> int:
    return max(1, image.shape[1] // 640)  # line widths / fonts grow with resolution


def draw_label(image: np.ndarray, text: str, org: tuple[int, int], color=WHITE, bg=(0, 0, 0)) -> None:
    s = _scale(image)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, max(1, s)
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    x, y = org
    y = max(y, tht + base + 2)
    cv2.rectangle(image, (x, y - tht - base - 2), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(image, text, (x + 2, y - base), font, fs, color, th, cv2.LINE_AA)


def draw_gaze(image: np.ndarray, frame: GazeFrame) -> np.ndarray:
    """Draw trail and gaze dot (hollow while blinking) in place."""
    h, w = image.shape[:2]
    radius = max(6, w // 80)
    pts = [(int(u), int(v)) for u, v in frame.trail if in_image((u, v), w, h)]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(image, a, b, TRAIL_COLOR, max(2, radius // 3), cv2.LINE_AA)
    if frame.gaze_px is not None:
        center = (int(frame.gaze_px[0]), int(frame.gaze_px[1]))
        if frame.blinking:
            cv2.circle(image, center, radius, DOT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, center, radius, DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, center, radius * 3, DOT_COLOR, 2, cv2.LINE_AA)
    return image


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    attended: Detection | None = None,
    dwell_s: float = 0.0,
) -> np.ndarray:
    """Thin grey boxes for all detections, a thick green box + dwell for the attended one."""
    s = _scale(image)
    for d in detections:
        if attended is not None and d is attended:
            continue
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, s, cv2.LINE_AA)
        draw_label(image, f"{d.class_name} {d.confidence:.2f}", (x1, y1), color=WHITE, bg=(70, 70, 70))
    if attended is not None:
        x1, y1, x2, y2 = (int(v) for v in attended.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), ATTENDED_COLOR, 3 * s, cv2.LINE_AA)
        draw_label(
            image, f"{attended.class_name}  {dwell_s:.1f}s", (x1, y1), color=(0, 0, 0), bg=ATTENDED_COLOR
        )
    return image


def draw_status(image: np.ndarray, frame: GazeFrame, extra: str = "") -> np.ndarray:
    text = (
        f"t={frame.t_rel_s:6.2f}s" + ("  BLINK" if frame.blinking else "") + (f"  {extra}" if extra else "")
    )
    draw_label(image, text, (8, 8))
    return image


def render(
    samples: Iterable[Sample],
    projector: GazeProjector,
    trail_s: float = 0.5,
    stats: OverlayStats | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """M1 checkpoint: (t_ns, image with gaze overlay) for every RGB frame."""
    for frame in gaze_frames(samples, projector, trail_s, stats):
        yield frame.t_ns, draw_status(draw_gaze(frame.image, frame), frame)


def write_video(
    frames: Iterable[tuple[int, np.ndarray]], out: Path, show: bool = False, scale: float = 1.0
) -> int:
    """Write RGB frames to an MP4 (fps estimated from timestamps); optionally show them live."""
    writer: cv2.VideoWriter | None = None
    pending: list[tuple[int, np.ndarray]] = []
    written = 0

    def open_writer(t0: int, t1: int, shape: tuple[int, ...]) -> cv2.VideoWriter:
        fps = 1e9 / (t1 - t0) if t1 > t0 else 30.0
        out.parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shape[1], shape[0]))

    try:
        for t_ns, frame_rgb in frames:
            if scale != 1.0:
                frame_rgb = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if show:
                cv2.imshow("aria-demo (q to quit)", bgr)
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
