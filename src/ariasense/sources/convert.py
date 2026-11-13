"""SDK data objects -> our typed samples. Shared by VRS replay and live streaming."""

from __future__ import annotations

from typing import Any

import numpy as np

from ariasense.types import AlsSample, EyeGazeSample, ImuSample, PpgSample, RgbFrame


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) and x > 0 else None


def rgb_frame(image_data: Any, t_ns: int) -> RgbFrame | None:
    if hasattr(image_data, "is_valid") and not image_data.is_valid():
        return None
    image = image_data.to_numpy_array()
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:  # grayscale fallback
        image = np.repeat(image[..., None], 3, axis=2)
    return RgbFrame(t_ns=t_ns, image=image)


def eye_gaze_sample(g: Any, t_ns: int) -> EyeGazeSample:
    v = getattr(g, "vergence", None)
    return EyeGazeSample(
        t_ns=t_ns,
        yaw=float(g.yaw),
        pitch=float(g.pitch),
        depth_m=_finite_or_none(g.depth),
        valid=bool(getattr(g, "combined_gaze_valid", True)),
        # Gen 2 on-device flags are True while the eye is OPEN (verified on s01_desk:
        # True ~99% of samples, False in 3-4 sample bursts = blinks), so invert them.
        left_blink=(not v.left_blink) if v is not None and v.left_blink_valid else None,
        right_blink=(not v.right_blink) if v is not None and v.right_blink_valid else None,
    )


def eye_gaze_time_ns(g: Any) -> int:
    """Device time of a streamed EyeGaze (its `tracking_timestamp` is a timedelta)."""
    ts = g.tracking_timestamp
    return round(ts.total_seconds() * 1e9) if hasattr(ts, "total_seconds") else int(ts)


def ppg_sample(p: Any, t_ns: int) -> PpgSample:
    return PpgSample(t_ns=t_ns, value=float(p.value))


def als_sample(a: Any, t_ns: int) -> AlsSample:
    return AlsSample(t_ns=t_ns, lux=float(a.lux))


def imu_sample(m: Any, label: str, t_ns: int) -> ImuSample:
    return ImuSample(
        t_ns=t_ns,
        label=label,
        accel=np.asarray(m.accel_msec2, dtype=float),
        gyro=np.asarray(m.gyro_radsec, dtype=float),
    )
