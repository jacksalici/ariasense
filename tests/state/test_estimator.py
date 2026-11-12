import pytest

from aria_demo.state.baseline import Baseline
from aria_demo.state.config import StateConfig
from aria_demo.state.estimator import State, StateEstimator
from aria_demo.state.features import FeatureVector

SEC = 1_000_000_000

BASE = {
    "hr_bpm": 70.0,
    "hr_fast_bpm": 70.0,
    "rmssd_ms": 60.0,
    "head_gyro_rms": 0.2,
    "head_gyro_peak": 0.5,
    "head_jerk": 5.0,
    "fix_dur_ms": 200.0,
    "fix_ratio": 0.6,
    "gaze_disp_deg": 10.0,
    "blink_rate_pm": 12.0,
    "blink_dur_ms": 150.0,
    "perclos": 0.03,
}


def make_baseline() -> Baseline:
    # Identical rows: every std falls back to its MIN_STD floor, so z = change / floor.
    rows = [FeatureVector(i * SEC, dict(BASE)) for i in range(10)]
    return Baseline.from_vectors(rows)


def fv(t_s: float, **changes: float) -> FeatureVector:
    v = dict(BASE)
    v.update(changes)
    return FeatureVector(int(t_s * SEC), v)


def run(est: StateEstimator, seconds: range, **changes: float):
    return [est.update(fv(t, **changes)) for t in seconds]


def cfg(**overrides) -> StateConfig:
    c = StateConfig()
    c.smoothing_s = 0.0  # no EMA lag in tests
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def test_calibrating_without_baseline():
    est = StateEstimator(cfg(), baseline=None)
    assert est.update(fv(0)).state is State.UNKNOWN


def test_calm_at_baseline_values():
    out = run(StateEstimator(cfg(), make_baseline()), range(5))
    assert out[-1].state is State.CALM
    assert out[-1].arousal == pytest.approx(0)


def test_focused_with_reasons():
    est = StateEstimator(cfg(), make_baseline())
    run(est, range(0, 5))
    out = run(est, range(5, 12), fix_dur_ms=400.0, gaze_disp_deg=4.0, fix_ratio=0.85)
    assert out[0].state is State.CALM  # hysteresis: not switched yet
    assert out[0].candidate is State.FOCUSED
    assert out[-1].state is State.FOCUSED
    assert out[-1].confidence > 0.8
    assert any(r.startswith("gaze spread -6.0°") for r in out[-1].reasons)
    assert any(r.startswith("fixations +200 ms") for r in out[-1].reasons)


def test_distracted():
    est = StateEstimator(cfg(), make_baseline())
    out = run(est, range(10), fix_dur_ms=100.0, gaze_disp_deg=16.0, fix_ratio=0.3)
    assert out[-1].state is State.DISTRACTED


def test_stressed_needs_sustained_arousal():
    est = StateEstimator(cfg(), make_baseline())
    out = run(est, range(30), hr_bpm=80.0, rmssd_ms=40.0)
    first_stressed = next(i for i, e in enumerate(out) if e.state is State.STRESSED)
    assert first_stressed >= 15  # needs ~75% of the 20 s window + hysteresis
    assert any(r.startswith("HR +10 bpm") for r in out[-1].reasons)


def test_fatigued_from_perclos():
    est = StateEstimator(cfg(), make_baseline())
    out = run(est, range(10), perclos=0.12, blink_dur_ms=300.0)
    assert out[-1].state is State.FATIGUED
    assert any("eyes closed" in r for r in out[-1].reasons)


def test_startle_needs_motion_spike_then_sustained_hr_rise():
    est = StateEstimator(cfg(), make_baseline())
    run(est, range(0, 10))
    assert est.update(fv(10, head_jerk=30.0)).state is not State.STARTLED  # spike only
    assert est.update(fv(11, hr_fast_bpm=80.0)).state is not State.STARTLED  # 1st rise
    e = est.update(fv(12, hr_fast_bpm=81.0))  # 2nd consecutive rise -> confirmed
    assert e.state is State.STARTLED
    assert e.reasons[0].startswith("head jerk") and "HR +11 bpm" in e.reasons[1]
    held = run(est, range(13, 18))
    assert held[-1].state is State.STARTLED  # held for 5 s
    assert est.update(fv(19)).state is not State.STARTLED


def test_no_startle_from_hr_rise_alone_or_late_rise():
    est = StateEstimator(cfg(), make_baseline())
    run(est, range(0, 10))
    out = run(est, range(10, 14), hr_fast_bpm=85.0)
    assert all(e.state is not State.STARTLED for e in out)

    est = StateEstimator(cfg(), make_baseline())
    run(est, range(0, 10))
    est.update(fv(10, head_jerk=30.0))
    out = run(est, range(17, 21), hr_fast_bpm=85.0)  # after the 6 s window
    assert all(e.state is not State.STARTLED for e in out)


def test_extras_feed_attention():
    est = StateEstimator(cfg(), make_baseline())
    without = est.update(fv(0)).attention
    est = StateEstimator(cfg(), make_baseline())
    with_dwell = est.update(fv(0), extras={"object_dwell_s": 2.0}).attention
    assert with_dwell > without


def test_config_loads_yaml(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("min_state_s: 7\nthresholds:\n  focused_attention: 1.1\n")
    c = StateConfig.load(p)
    assert c.min_state_s == 7 and c.thresholds.focused_attention == 1.1
    assert c.startle.motion_z == 4.0  # untouched defaults
    assert StateConfig.load(tmp_path / "missing.yaml") == StateConfig()
