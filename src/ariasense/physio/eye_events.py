"""Blinks, fixations and gaze dispersion from the on-device eye gaze (~30 Hz).

Gaze angles are head-relative, so with head motion the eyes counter-rotate
(VOR) and fixations on a world object can look like slow drifts. The I-DT
thresholds are generous to tolerate that at a desk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ariasense.types import EyeGazeSample

# On s01_desk the 30 Hz on-device gaze jitters ~0.8° between consecutive samples
# (median), so lab-style 1–2° thresholds split fixations apart. At 4° the mean
# fixation is ~240 ms with ~70% of the time fixating, in line with normal viewing.
FIX_DISPERSION_DEG = 4.0
FIX_MIN_DURATION_S = 0.1


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class EyeStats:
    blink_rate_pm: float  # blinks per minute
    blink_dur_ms: float | None  # mean blink duration
    perclos: float  # fraction of time with eyes closed
    fix_dur_ms: float | None  # mean fixation duration
    fix_rate_hz: float  # fixations per second (≈ saccade rate)
    fix_ratio: float  # fraction of time spent in fixations
    gaze_disp_deg: float | None  # spread of gaze direction (RMS distance from mean)
    valid_ratio: float = 1.0  # fraction of samples with valid gaze (~0 when not worn)


def eyes_closed(samples: Sequence[EyeGazeSample]) -> np.ndarray:
    """Closed when both eyes report a blink; with one eye unknown, trust the other."""
    out = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        flags = [f for f in (s.left_blink, s.right_blink) if f is not None]
        out[i] = bool(flags) and all(flags)
    return out


def runs(t_s: np.ndarray, mask: np.ndarray) -> list[Interval]:
    """Contiguous True runs as intervals (end = time of the first sample after the run)."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append(Interval(t_s[start], t_s[i]))
            start = None
    if start is not None:
        out.append(Interval(t_s[start], t_s[-1]))
    return out


def idt_fixations(
    t_s: np.ndarray,
    yaw_deg: np.ndarray,
    pitch_deg: np.ndarray,
    usable: np.ndarray,
    dispersion_deg: float = FIX_DISPERSION_DEG,
    min_duration_s: float = FIX_MIN_DURATION_S,
) -> list[Interval]:
    """Dispersion-threshold (I-DT) fixation detection; unusable samples end a fixation."""

    def dispersion(a: int, b: int) -> float:
        y, p = yaw_deg[a : b + 1], pitch_deg[a : b + 1]
        return float((y.max() - y.min()) + (p.max() - p.min()))

    fixations: list[Interval] = []
    n, i = len(t_s), 0
    while i < n:
        if not usable[i]:
            i += 1
            continue
        # Smallest window starting at i that spans the minimum duration.
        j = i
        while j + 1 < n and usable[j + 1] and t_s[j] - t_s[i] < min_duration_s:
            j += 1
        if t_s[j] - t_s[i] < min_duration_s:
            i = j + 1  # interrupted by unusable samples or the end
            continue
        if dispersion(i, j) > dispersion_deg:
            i += 1
            continue
        while j + 1 < n and usable[j + 1] and dispersion(i, j + 1) <= dispersion_deg:
            j += 1
        fixations.append(Interval(t_s[i], t_s[j]))
        i = j + 1
    return fixations


def eye_stats(samples: Sequence[EyeGazeSample], window_s: float | None = None) -> EyeStats | None:
    """Statistics over `samples` (expected to span about `window_s` seconds)."""
    if len(samples) < 3:
        return None
    t = np.array([s.t_ns for s in samples], dtype=float) / 1e9
    span = window_s or float(t[-1] - t[0])
    if span <= 0:
        return None

    closed = eyes_closed(samples)
    blinks = runs(t, closed)
    blink_dur = [b.duration_s for b in blinks if b.duration_s > 0]

    yaw = np.degrees([s.yaw for s in samples])
    pitch = np.degrees([s.pitch for s in samples])
    usable = np.array([s.valid for s in samples]) & ~closed
    fixations = idt_fixations(t, yaw, pitch, usable)
    fix_time = sum(f.duration_s for f in fixations)

    disp = None
    if usable.sum() >= 3:
        y, p = yaw[usable], pitch[usable]
        disp = float(np.sqrt(np.mean((y - y.mean()) ** 2 + (p - p.mean()) ** 2)))

    return EyeStats(
        blink_rate_pm=60.0 * len(blinks) / span,
        blink_dur_ms=1000.0 * float(np.mean(blink_dur)) if blink_dur else None,
        perclos=float(closed.mean()),
        fix_dur_ms=1000.0 * fix_time / len(fixations) if fixations else None,
        fix_rate_hz=len(fixations) / span,
        fix_ratio=min(1.0, fix_time / span),
        gaze_disp_deg=disp,
        valid_ratio=float(np.mean([s.valid for s in samples])),
    )
