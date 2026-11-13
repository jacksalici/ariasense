import numpy as np
import pytest

from ariasense.physio.ppg import FS_HZ, estimate_heart, valid_ibi_mask


def synthetic_ppg(hr_bpm=72.0, seconds=30.0, jitter_s=0.0, spikes=True, invert=False, seed=0):
    """Pulse train with per-beat timing jitter, baseline drift, noise and spikes."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FS_HZ)
    beats, b = [], 0.0
    while b < seconds:
        beats.append(b)
        b += 60 / hr_bpm + rng.normal(0, jitter_s)
    x = np.zeros_like(t)
    for bt in beats:
        x += np.exp(-0.5 * ((t - bt) / 0.08) ** 2)  # systolic pulse
    x += 0.3 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.03, t.size)  # drift + noise
    if spikes:
        idx = rng.choice(t.size, 40, replace=False)
        x[idx] += rng.choice([-3, 3], idx.size)  # single-sample spikes like the real sensor
    x = 166_000 + 800 * (-x if invert else x)
    return t, x, np.diff(beats)


@pytest.mark.parametrize("hr", [55.0, 72.0, 110.0])
@pytest.mark.parametrize("invert", [False, True])
def test_heart_rate_recovered(hr, invert):
    t, x, _ = synthetic_ppg(hr_bpm=hr, invert=invert)
    est = estimate_heart(t, x)
    assert est.hr_bpm == pytest.approx(hr, rel=0.03)
    assert est.quality > 0.9


def test_rmssd_tracks_beat_jitter():
    t, x, ibi = synthetic_ppg(hr_bpm=65, jitter_s=0.04, seed=3)
    truth = 1000 * np.sqrt(np.mean(np.diff(ibi) ** 2))
    est = estimate_heart(t, x)
    assert est.rmssd_ms == pytest.approx(truth, rel=0.25)

    steady = estimate_heart(*synthetic_ppg(hr_bpm=65, jitter_s=0.0)[:2])
    assert steady.rmssd_ms < est.rmssd_ms / 3


def test_too_short_or_flat_signal_gives_no_estimate():
    t, x, _ = synthetic_ppg(seconds=3)
    assert estimate_heart(t, x).hr_bpm is None
    t = np.arange(0, 20, 1 / FS_HZ)
    assert estimate_heart(t, np.full_like(t, 166_000.0)).hr_bpm is None


def test_ibi_artefacts_rejected():
    ibi = np.array([0.85, 0.86, 0.84, 0.43, 0.87, 0.85, 1.9, 0.86])
    assert valid_ibi_mask(ibi).tolist() == [True, True, True, False, True, True, False, True]
