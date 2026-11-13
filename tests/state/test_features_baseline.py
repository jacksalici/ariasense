import numpy as np
import pytest

from ariasense.state.baseline import MIN_STD, Baseline
from ariasense.state.features import FEATURES, FeatureExtractor, FeatureVector, extract
from ariasense.types import AlsSample, EyeGazeSample, ImuSample, PpgSample

SEC = 1_000_000_000


def stream(seconds=12):
    """Interleaved, time-ordered synthetic samples: PPG 128 Hz, gaze 30 Hz, IMU 800 Hz, ALS 10 Hz."""
    out = []
    t = np.arange(0, seconds, 1 / 128)
    ppg = 166_000 + 800 * np.exp(-0.5 * (((t % (60 / 70)) - 0.2) / 0.08) ** 2)
    out += [PpgSample(int(ti * SEC), float(v)) for ti, v in zip(t, ppg, strict=True)]
    out += [
        EyeGazeSample(int(ti * SEC), 0.0, 0.0, None, True, False, False)
        for ti in np.arange(0, seconds, 1 / 30)
    ]
    out += [
        ImuSample(int(ti * SEC), lbl, np.array([0, 0, 9.81]), np.array([0, 0, 0.2]))
        for ti in np.arange(0, seconds, 1 / 800)
        for lbl in ("imu-left", "imu-right")
    ]
    out += [AlsSample(int(ti * SEC), 50.0) for ti in np.arange(0, seconds, 1 / 10)]
    return sorted(out, key=lambda s: s.t_ns)


def test_extractor_emits_once_per_step_with_all_groups():
    rows = list(extract(stream(12), FeatureExtractor(step_s=1.0)))
    assert len(rows) == 11
    assert all(set(r.values) == set(FEATURES) for r in rows)
    last = rows[-1]
    assert last.get("hr_bpm") == pytest.approx(70, rel=0.03)
    assert last.get("blink_rate_pm") == 0
    assert last.get("head_gyro_rms") == pytest.approx(0.2)
    assert last.get("lux") == 50.0
    assert rows[0].get("hr_bpm") is None  # not enough PPG yet


def test_baseline_freezes_after_duration_and_z_scores():
    b = Baseline(duration_s=3, min_samples=3)
    for i, hr in enumerate([60, 62, 64, 62]):
        assert not b.add(FeatureVector(i * SEC, {"hr_bpm": float(hr), "lux": None}))
    assert b.add(FeatureVector(4 * SEC, {"hr_bpm": 99.0}))  # past 3 s: frozen, not included
    assert b.mean["hr_bpm"] == pytest.approx(62)
    assert b.std["hr_bpm"] == pytest.approx(max(np.std([60, 62, 64, 62]), MIN_STD["hr_bpm"]))
    assert "lux" not in b.mean  # never observed
    z = b.z(FeatureVector(5 * SEC, {"hr_bpm": 62 + 2 * b.std["hr_bpm"], "lux": 10.0}))
    assert z == {"hr_bpm": pytest.approx(2.0), "lux": None}


def test_baseline_std_floor():
    b = Baseline.from_vectors([FeatureVector(i, {"hr_bpm": 70.0}) for i in range(10)])
    assert b.std["hr_bpm"] == MIN_STD["hr_bpm"]
