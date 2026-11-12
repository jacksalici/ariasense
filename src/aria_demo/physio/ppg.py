"""Heart rate and HRV from the raw Aria Gen 2 PPG (nose pad, ~128 Hz).

Steps (tuned on s01_desk):
1. Median filter removes the single-sample spikes in the raw signal.
2. Band-pass 0.7–3.5 Hz (42–210 bpm).
3. Peak detection on both polarities, keeping the one with the more regular
   inter-beat intervals (IBIs).
4. IBIs outside 0.33–1.5 s, or >20% off the local median, count as artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

FS_HZ = 128.0
BAND_HZ = (0.7, 3.5)
IBI_RANGE_S = (0.33, 1.5)
IBI_MAX_DEV = 0.2
# Below this band-passed std (raw sensor units) there is no pulse to measure,
# e.g. the nose pad is not touching the skin. On s01_desk the std is ~200.
MIN_PULSE_STD = 5.0


@dataclass(frozen=True)
class HeartEstimate:
    hr_bpm: float | None
    rmssd_ms: float | None
    quality: float  # fraction of IBIs kept after artefact rejection (0–1)
    n_beats: int
    hr_fast_bpm: float | None = None  # last 3 valid beats: reacts within seconds (startle)


NO_ESTIMATE = HeartEstimate(None, None, 0.0, 0)
FAST_BEATS = 3


def filter_ppg(values: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    """Despike and band-pass. Needs a few seconds of signal."""
    x = signal.medfilt(np.asarray(values, dtype=float), 7)
    sos = signal.butter(3, BAND_HZ, btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _peaks(y: np.ndarray, fs: float) -> np.ndarray:
    prominence = 0.5 * float(np.std(y))
    idx, _ = signal.find_peaks(y, distance=max(1, int(fs * IBI_RANGE_S[0])), prominence=prominence)
    return idx


def detect_beats(t_s: np.ndarray, y: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    """Beat times (s). Polarity is chosen by IBI regularity (lowest coefficient of variation)."""
    best, best_cv = np.array([]), np.inf
    for sign in (1.0, -1.0):
        beats = t_s[_peaks(sign * y, fs)]
        if len(beats) < 3:
            continue
        ibi = np.diff(beats)
        cv = float(np.std(ibi) / np.mean(ibi))
        if cv < best_cv:
            best, best_cv = beats, cv
    return best


def valid_ibi_mask(ibi: np.ndarray) -> np.ndarray:
    """True for physiologically plausible IBIs that agree with their neighbours."""
    if len(ibi) == 0:
        return np.zeros(0, dtype=bool)
    in_range = (ibi >= IBI_RANGE_S[0]) & (ibi <= IBI_RANGE_S[1])
    local = signal.medfilt(ibi, 5) if len(ibi) >= 5 else np.full_like(ibi, np.median(ibi))
    return in_range & (np.abs(ibi - local) <= IBI_MAX_DEV * local)


def estimate_heart(
    t_s: np.ndarray,
    values: np.ndarray,
    fs: float = FS_HZ,
    hr_window_s: float = 10.0,
    min_signal_s: float = 6.0,
) -> HeartEstimate:
    """HR from the last `hr_window_s`, RMSSD over the whole given signal."""
    t_s = np.asarray(t_s, dtype=float)
    if len(t_s) < 2 or t_s[-1] - t_s[0] < min_signal_s:
        return NO_ESTIMATE
    y = filter_ppg(values, fs)
    if float(np.std(y)) < MIN_PULSE_STD:
        return NO_ESTIMATE
    beats = detect_beats(t_s, y, fs)
    if len(beats) < 3:
        return NO_ESTIMATE

    ibi = np.diff(beats)
    ok = valid_ibi_mask(ibi)
    quality = float(ok.mean())

    recent = ok & (beats[1:] >= t_s[-1] - hr_window_s)
    hr = 60.0 / float(np.median(ibi[recent])) if recent.sum() >= 2 else None
    last_ok = ibi[ok][-FAST_BEATS:]
    hr_fast = 60.0 / float(np.mean(last_ok)) if len(last_ok) == FAST_BEATS else None

    successive = ok[1:] & ok[:-1]
    diffs = np.diff(ibi)[successive]
    rmssd = 1000.0 * float(np.sqrt(np.mean(diffs**2))) if len(diffs) >= 5 else None
    return HeartEstimate(hr, rmssd, quality, len(beats), hr_fast)
