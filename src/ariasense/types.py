"""Typed, SDK-independent samples that flow through the pipeline.

Every sample carries `t_ns`, its device-time timestamp in nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RgbFrame:
    t_ns: int
    image: np.ndarray  # H x W x 3, uint8, RGB, in the camera's raw (sideways) orientation


@dataclass(frozen=True, slots=True)
class EyeGazeSample:
    t_ns: int
    yaw: float  # rad, in the central pupil frame (CPF)
    pitch: float  # rad
    depth_m: float | None  # vergence depth; None when not available
    valid: bool
    left_blink: bool | None = None
    right_blink: bool | None = None


@dataclass(frozen=True, slots=True)
class PpgSample:
    t_ns: int
    value: float  # raw PPG reading


@dataclass(frozen=True, slots=True)
class AlsSample:
    t_ns: int
    lux: float


@dataclass(frozen=True, slots=True)
class ImuSample:
    t_ns: int
    label: str  # e.g. "imu-left"; Gen 2 has more than one IMU
    accel: np.ndarray  # m/s^2, shape (3,)
    gyro: np.ndarray  # rad/s, shape (3,)


Sample = RgbFrame | EyeGazeSample | PpgSample | AlsSample | ImuSample


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in upright-image pixels
    track_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
