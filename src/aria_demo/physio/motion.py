"""Head motion from one IMU: rotation speed and jerk (sudden movements, e.g. a startle)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from aria_demo.types import ImuSample


@dataclass(frozen=True)
class MotionStats:
    gyro_rms: float  # rad/s, overall head rotation speed
    gyro_peak: float  # rad/s, fastest rotation in the window
    jerk_rms: float  # m/s^3, derivative of acceleration


def motion_stats(samples: Sequence[ImuSample]) -> MotionStats | None:
    if len(samples) < 3:
        return None
    t = np.array([s.t_ns for s in samples], dtype=float) / 1e9
    gyro = np.linalg.norm(np.stack([s.gyro for s in samples]), axis=1)
    accel = np.stack([s.accel for s in samples])
    dt = np.diff(t)
    # Smooth acceleration over ~25 ms first: differentiating raw IMU noise explodes.
    size = max(1, int(round(0.025 / float(np.median(dt))))) if len(dt) else 1
    accel = uniform_filter1d(accel, size=size, axis=0, mode="nearest")
    dt[dt <= 0] = np.nan
    jerk = np.linalg.norm(np.diff(accel, axis=0), axis=1) / dt
    return MotionStats(
        gyro_rms=float(np.sqrt(np.mean(gyro**2))),
        gyro_peak=float(gyro.max()),
        jerk_rms=float(np.sqrt(np.nanmean(jerk**2))),
    )
