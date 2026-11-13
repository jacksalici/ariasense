import numpy as np
import pytest

from ariasense.physio.eye_events import eye_stats, eyes_closed, idt_fixations, runs
from ariasense.physio.motion import motion_stats
from ariasense.types import EyeGazeSample, ImuSample

HZ = 30


def gaze_seq(yaw_deg, closed=None):
    closed = closed if closed is not None else [False] * len(yaw_deg)
    return [
        EyeGazeSample(
            t_ns=int(i * 1e9 / HZ),
            yaw=float(np.radians(y)),
            pitch=0.0,
            depth_m=None,
            valid=True,
            left_blink=c,
            right_blink=c,
        )
        for i, (y, c) in enumerate(zip(yaw_deg, closed, strict=True))
    ]


def test_runs_and_closed():
    t = np.arange(6.0)
    assert [(r.start_s, r.end_s) for r in runs(t, np.array([0, 1, 1, 0, 0, 1], bool))] == [(1, 3), (5, 5)]
    s = EyeGazeSample(0, 0, 0, None, True, left_blink=True, right_blink=None)
    assert eyes_closed([s]).tolist() == [True]  # unknown eye ignored
    s = EyeGazeSample(0, 0, 0, None, True, left_blink=True, right_blink=False)
    assert eyes_closed([s]).tolist() == [False]  # both must be closed


def test_idt_finds_two_fixations_separated_by_a_saccade():
    yaw = np.r_[np.full(15, 0.0), np.full(15, 20.0)]  # 0.5 s at 0°, 0.5 s at 20°
    t = np.arange(yaw.size) / HZ
    fix = idt_fixations(t, yaw, np.zeros_like(yaw), np.ones_like(yaw, bool))
    assert len(fix) == 2
    assert fix[0].duration_s == pytest.approx(14 / HZ)


def test_idt_rejects_smooth_sweep_and_respects_unusable_samples():
    yaw = np.linspace(0, 60, 30)  # 2°/sample sweep: never stable
    t = np.arange(30) / HZ
    assert idt_fixations(t, yaw, np.zeros(30), np.ones(30, bool)) == []
    usable = np.ones(30, bool)
    usable[::2] = False  # never two usable samples in a row
    assert idt_fixations(t, np.zeros(30), np.zeros(30), usable) == []


def test_eye_stats_blinks_and_fixations():
    n = HZ * 10
    closed = np.zeros(n, bool)
    for start in (30, 120, 210):  # three 4-sample blinks in 10 s
        closed[start : start + 4] = True
    yaw = np.where(np.arange(n) < n // 2, 0.0, 15.0)
    st = eye_stats(gaze_seq(yaw, closed.tolist()), window_s=10)
    assert st.blink_rate_pm == pytest.approx(18)
    assert st.blink_dur_ms == pytest.approx(4000 / HZ)
    assert st.perclos == pytest.approx(12 / n)
    assert st.fix_ratio > 0.9
    assert eye_stats(gaze_seq([0.0, 0.0])) is None


def test_motion_stats_detects_rotation_and_jerk():
    def imu(gyro_z, accel_x):
        return [
            ImuSample(int(i * 5e6), "imu-left", np.array([a, 0, 9.81]), np.array([0, 0, g]))
            for i, (g, a) in enumerate(zip(gyro_z, accel_x, strict=True))
        ]

    still = motion_stats(imu(np.zeros(400), np.zeros(400)))
    assert still.gyro_rms == 0 and still.jerk_rms == pytest.approx(0)

    accel = np.zeros(400)
    accel[200:] = 5.0  # sudden 5 m/s^2 step
    moving = motion_stats(imu(np.full(400, 1.0), accel))
    assert moving.gyro_rms == pytest.approx(1.0)
    assert moving.jerk_rms > 10
