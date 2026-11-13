"""Personal baseline: mean/std of each feature over a calm period, then z-scores."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ariasense.state.features import FeatureVector

# Minimum std per feature, so a very steady baseline does not turn noise into huge z-scores.
MIN_STD: dict[str, float] = {
    "hr_bpm": 2.0,
    "hr_fast_bpm": 3.0,
    "rmssd_ms": 5.0,
    "ppg_quality": 0.05,
    "blink_rate_pm": 2.0,
    "blink_dur_ms": 20.0,
    "perclos": 0.01,
    "fix_dur_ms": 50.0,
    "fix_rate_hz": 0.2,
    "fix_ratio": 0.05,
    "gaze_disp_deg": 1.0,
    "gaze_valid": 0.05,
    "head_gyro_rms": 0.05,
    "head_gyro_peak": 0.1,
    "head_jerk": 2.0,
    "lux": 5.0,
}


class Baseline:
    """Collects feature vectors for `duration_s` after the first one, then freezes."""

    def __init__(self, duration_s: float = 60.0, min_samples: int = 5) -> None:
        self.duration_ns = int(duration_s * 1e9)
        self.min_samples = min_samples
        self._t0: int | None = None
        self._rows: list[FeatureVector] = []
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.ready = False

    def add(self, fv: FeatureVector) -> bool:
        """Feed a vector; returns True once the baseline is frozen."""
        if self.ready:
            return True
        self._t0 = fv.t_ns if self._t0 is None else self._t0
        if fv.t_ns - self._t0 <= self.duration_ns:
            self._rows.append(fv)
            return False
        self.freeze()
        return True

    def freeze(self) -> None:
        names = {k for fv in self._rows for k in fv.values}
        for name in names:
            vals = np.array([v for fv in self._rows if (v := fv.get(name)) is not None], dtype=float)
            if len(vals) >= self.min_samples:
                self.mean[name] = float(vals.mean())
                self.std[name] = max(float(vals.std()), MIN_STD.get(name, 1e-6))
        self.ready = True

    def z(self, fv: FeatureVector) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, value in fv.values.items():
            if value is None or name not in self.mean:
                out[name] = None
            else:
                out[name] = (value - self.mean[name]) / self.std[name]
        return out

    @classmethod
    def from_vectors(cls, vectors: Iterable[FeatureVector], min_samples: int = 5) -> Baseline:
        b = cls(duration_s=0.0, min_samples=min_samples)
        b._rows = list(vectors)
        b.freeze()
        return b
