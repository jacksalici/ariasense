"""Source protocol: anything that yields time-ordered samples plus a calibration."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Any, Protocol

from ariasense.types import Sample


class Signal(StrEnum):
    RGB = "rgb"
    EYE_GAZE = "eye_gaze"
    PPG = "ppg"
    ALS = "als"
    IMU = "imu"


ALL_SIGNALS = frozenset(Signal)


class Source(Protocol):
    def device_calibration(self) -> Any:
        """projectaria_tools DeviceCalibration for the glasses that produced the data."""
        ...

    def samples(self) -> Iterator[Sample]:
        """Samples in device-time order."""
        ...


class Pacer:
    """Sleeps so that samples are released at `speed` x real time.

    speed <= 0 disables pacing (as fast as possible).
    """

    def __init__(
        self,
        speed: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.speed = speed
        self._clock = clock
        self._sleep = sleep
        self._t0_ns: int | None = None
        self._wall0 = 0.0

    def rebase(self, speed: float | None = None) -> None:
        """Restart pacing from the next sample (after a pause or a speed change)."""
        if speed is not None:
            self.speed = speed
        self._t0_ns = None

    def wait_until(self, t_ns: int) -> None:
        if self.speed <= 0:
            return
        if self._t0_ns is None:
            self._t0_ns, self._wall0 = t_ns, self._clock()
            return
        due = self._wall0 + (t_ns - self._t0_ns) / 1e9 / self.speed
        delay = due - self._clock()
        if delay > 0:
            self._sleep(delay)
