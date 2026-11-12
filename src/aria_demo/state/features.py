"""Turn the raw sample stream into a feature vector every `step_s` of device time."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from aria_demo.physio.eye_events import eye_stats
from aria_demo.physio.motion import motion_stats
from aria_demo.physio.ppg import estimate_heart
from aria_demo.sync import History
from aria_demo.types import AlsSample, EyeGazeSample, ImuSample, PpgSample, Sample

# Feature name -> human-readable description (and the column order of reports).
FEATURES: dict[str, str] = {
    "hr_bpm": "Heart rate (bpm)",
    "hr_fast_bpm": "Instant heart rate, last 3 beats (bpm)",
    "rmssd_ms": "HRV, RMSSD (ms)",
    "ppg_quality": "PPG beat quality (0-1)",
    "blink_rate_pm": "Blinks per minute",
    "blink_dur_ms": "Mean blink duration (ms)",
    "perclos": "Fraction of time eyes closed",
    "fix_dur_ms": "Mean fixation duration (ms)",
    "fix_rate_hz": "Fixations per second",
    "fix_ratio": "Fraction of time fixating",
    "gaze_disp_deg": "Gaze dispersion (deg)",
    "gaze_valid": "Fraction of valid gaze samples",
    "head_gyro_rms": "Head rotation speed (rad/s)",
    "head_gyro_peak": "Peak head rotation (rad/s)",
    "head_jerk": "Head jerk (m/s^3)",
    "lux": "Ambient light (lux)",
}


@dataclass
class FeatureVector:
    t_ns: int
    values: dict[str, float | None] = field(default_factory=dict)

    def get(self, name: str) -> float | None:
        return self.values.get(name)


@dataclass(frozen=True)
class Windows:
    """Window lengths (s) per feature group. HRV needs a longer window than HR."""

    heart_s: float = 30.0
    hr_s: float = 10.0
    eye_s: float = 10.0
    motion_s: float = 2.0
    light_s: float = 2.0


class FeatureExtractor:
    """Buffers samples and emits a `FeatureVector` every `step_s` (device time)."""

    def __init__(self, windows: Windows = Windows(), step_s: float = 1.0, imu_decimate: int = 4) -> None:
        self.w = windows
        self.step_ns = int(step_s * 1e9)
        self.imu_decimate = imu_decimate
        sec = 1_000_000_000
        self._ppg: History[PpgSample] = History(int(windows.heart_s * sec))
        self._gaze: History[EyeGazeSample] = History(int(windows.eye_s * sec))
        self._imu: History[ImuSample] = History(int(windows.motion_s * sec))
        self._als: History[AlsSample] = History(int(windows.light_s * sec))
        self._imu_label: str | None = None
        self._imu_count = 0
        self._t0: int | None = None
        self._next_emit: int | None = None

    def add(self, sample: Sample) -> FeatureVector | None:
        """Add a sample; return a feature vector when a step boundary is crossed."""
        if isinstance(sample, PpgSample):
            self._ppg.add(sample)
        elif isinstance(sample, EyeGazeSample):
            self._gaze.add(sample)
        elif isinstance(sample, ImuSample):
            self._imu_label = self._imu_label or sample.label  # use a single IMU
            if sample.label == self._imu_label:
                self._imu_count += 1
                if self._imu_count % self.imu_decimate == 0:
                    self._imu.add(sample)
        elif isinstance(sample, AlsSample):
            self._als.add(sample)
        else:
            return None

        if self._t0 is None:
            self._t0, self._next_emit = sample.t_ns, sample.t_ns + self.step_ns
        if sample.t_ns >= self._next_emit:
            self._next_emit += self.step_ns * ((sample.t_ns - self._next_emit) // self.step_ns + 1)
            return self.compute(sample.t_ns)
        return None

    def compute(self, t_ns: int) -> FeatureVector:
        v: dict[str, float | None] = dict.fromkeys(FEATURES)

        ppg = self._ppg.since(t_ns - int(self.w.heart_s * 1e9))
        if ppg:
            t = np.array([s.t_ns for s in ppg], dtype=float) / 1e9
            heart = estimate_heart(t, np.array([s.value for s in ppg]), hr_window_s=self.w.hr_s)
            v["hr_bpm"], v["rmssd_ms"], v["ppg_quality"] = heart.hr_bpm, heart.rmssd_ms, heart.quality
            v["hr_fast_bpm"] = heart.hr_fast_bpm

        eye = eye_stats(self._gaze.since(t_ns - int(self.w.eye_s * 1e9)))
        if eye is not None:
            v.update(
                blink_rate_pm=eye.blink_rate_pm,
                blink_dur_ms=eye.blink_dur_ms,
                perclos=eye.perclos,
                fix_dur_ms=eye.fix_dur_ms,
                fix_rate_hz=eye.fix_rate_hz,
                fix_ratio=eye.fix_ratio,
                gaze_disp_deg=eye.gaze_disp_deg,
                gaze_valid=eye.valid_ratio,
            )

        motion = motion_stats(self._imu.since(t_ns - int(self.w.motion_s * 1e9)))
        if motion is not None:
            v["head_gyro_rms"], v["head_gyro_peak"], v["head_jerk"] = (
                motion.gyro_rms,
                motion.gyro_peak,
                motion.jerk_rms,
            )

        als = self._als.since(t_ns - int(self.w.light_s * 1e9))
        if als:
            v["lux"] = float(np.mean([s.lux for s in als]))
        return FeatureVector(t_ns, v)


def extract(samples: Iterable[Sample], extractor: FeatureExtractor | None = None) -> Iterator[FeatureVector]:
    extractor = extractor or FeatureExtractor()
    for sample in samples:
        fv = extractor.add(sample)
        if fv is not None:
            yield fv
