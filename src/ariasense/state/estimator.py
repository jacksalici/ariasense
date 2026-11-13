"""Explainable, rule-based estimate of the wearer's state.

1. Features are z-scored against the personal baseline and clipped.
2. Two axes: arousal (HR up, HRV down, ...) and attention (long fixations,
   narrow gaze, few blinks, ...), each a weighted mean of z-scores, EMA-smoothed.
3. Ordered rules map the axes (plus PERCLOS / blink length) to a candidate state.
4. Hysteresis: a candidate must persist `min_state_s` before it is shown.
5. Startle is an event: a head-motion spike followed within seconds by a fast
   heart-rate rise. It overrides the shown state for `hold_s`.

These are *estimated* physiological/attentional states, not measured emotions.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from ariasense.state.baseline import Baseline
from ariasense.state.config import StateConfig
from ariasense.state.features import FeatureVector


class State(StrEnum):
    UNKNOWN = "Calibrating"
    NO_SIGNAL = "No signal"  # glasses not worn: no valid gaze and no pulse
    NEUTRAL = "Neutral"
    CALM = "Calm"
    FOCUSED = "Focused"
    DISTRACTED = "Distracted"
    STRESSED = "Stressed"
    FATIGUED = "Fatigued"
    STARTLED = "Startled"


MIN_GAZE_VALID = 0.3


def not_worn_reasons(fv: FeatureVector) -> list[str]:
    """Why the glasses look unworn (empty list = worn).

    Worn means valid gaze for at least 30% of the eye window, or a pulse on the
    nose pad. Seen live with the glasses lying on a desk: gaze invalid, no pulse,
    yet every other feature looked "calm".
    """
    gaze_ok = (fv.get("gaze_valid") or 0.0) >= MIN_GAZE_VALID
    pulse_ok = fv.get("hr_bpm") is not None
    if gaze_ok or pulse_ok:
        return []
    return ["no valid eye gaze", "no pulse on the nose pad", "are the glasses being worn?"]


@dataclass
class StateEstimate:
    t_ns: int
    state: State
    confidence: float
    reasons: list[str]
    arousal: float | None
    attention: float | None
    candidate: State  # rule output before hysteresis / startle override
    z: dict[str, float] = field(default_factory=dict)


# Feature -> (label, unit format for the change vs baseline, multiplier)
_REASON_FORMAT: dict[str, tuple[str, str, float]] = {
    "hr_bpm": ("HR", "{:+.0f} bpm", 1.0),
    "hr_fast_bpm": ("HR", "{:+.0f} bpm", 1.0),
    "rmssd_ms": ("HRV", "{:+.0f} ms", 1.0),
    "head_gyro_rms": ("head motion", "{:+.2f} rad/s", 1.0),
    "head_jerk": ("head jerk", "{:+.0f} m/s³", 1.0),
    "head_gyro_peak": ("head turn", "{:+.1f} rad/s", 1.0),
    "fix_dur_ms": ("fixations", "{:+.0f} ms", 1.0),
    "fix_ratio": ("time fixating", "{:+.0f} pp", 100.0),
    "gaze_disp_deg": ("gaze spread", "{:+.1f}°", 1.0),
    "blink_rate_pm": ("blinks", "{:+.0f}/min", 1.0),
    "blink_dur_ms": ("blink length", "{:+.0f} ms", 1.0),
    "perclos": ("eyes closed", "{:+.1f} pp", 100.0),
}


def _confidence(margin: float) -> float:
    return 0.5 + 0.5 * math.tanh(max(margin, 0.0))


class StateEstimator:
    def __init__(self, config: StateConfig | None = None, baseline: Baseline | None = None) -> None:
        self.cfg = config or StateConfig()
        self.baseline = baseline
        self._last_t: int | None = None
        self._arousal: float | None = None
        self._attention: float | None = None
        self._arousal_hist: deque[tuple[int, float]] = deque()
        self._hr_fast_hist: deque[tuple[int, float]] = deque()
        self._shown = State.UNKNOWN
        self._pending: State | None = None
        self._pending_since = 0
        self._spike: tuple[int, float, float] | None = None  # (t, pre-spike HR, motion z)
        self._rise_count = 0  # consecutive estimates with the HR rise after a spike
        self._startle_until = -1
        self._startle_reasons: list[str] = []

    # ---- axes -------------------------------------------------------------
    def _zscores(self, fv: FeatureVector, extras: dict[str, float] | None) -> dict[str, float]:
        z = {k: v for k, v in (self.baseline.z(fv) if self.baseline else {}).items() if v is not None}
        z.update(extras or {})
        c = self.cfg.z_clip
        return {k: float(np.clip(v, -c, c)) for k, v in z.items()}

    @staticmethod
    def _axis(z: dict[str, float], weights: dict[str, float]) -> float | None:
        used = [(w, z[k]) for k, w in weights.items() if k in z and w != 0]
        if not used:
            return None
        return sum(w * v for w, v in used) / sum(abs(w) for w, _ in used)

    def _smooth(self, prev: float | None, new: float | None, dt_s: float) -> float | None:
        if new is None:
            return prev
        if prev is None:
            return new
        alpha = 1.0 - math.exp(-dt_s / self.cfg.smoothing_s) if self.cfg.smoothing_s > 0 else 1.0
        return prev + alpha * (new - prev)

    # ---- rules ------------------------------------------------------------
    def _candidate(self, z: dict[str, float], sustained: float | None) -> tuple[State, float]:
        th, a, t = self.cfg.thresholds, self._arousal, self._attention
        if self.baseline is None or not self.baseline.ready:
            return State.UNKNOWN, 0.0

        fatigue = max(
            z.get("perclos", -math.inf) - th.fatigue_perclos_z,
            z.get("blink_dur_ms", -math.inf) - th.fatigue_blink_dur_z,
        )
        if fatigue >= 0 and (a is None or a <= th.fatigue_max_arousal):
            return State.FATIGUED, fatigue
        if (
            sustained is not None
            and sustained >= th.stressed_arousal
            and (t is None or t < th.focused_attention)
        ):
            return State.STRESSED, sustained - th.stressed_arousal
        if t is not None and t >= th.focused_attention and (a is None or a <= th.focused_max_arousal):
            return State.FOCUSED, t - th.focused_attention
        if t is not None and t <= th.distracted_attention:
            return State.DISTRACTED, th.distracted_attention - t
        if a is not None and a <= th.calm_max_arousal:
            return State.CALM, th.calm_max_arousal - a
        return State.NEUTRAL, 0.0

    def _update_startle(self, t_ns: int, fv: FeatureVector, z: dict[str, float]) -> None:
        sc = self.cfg.startle
        hr_fast = fv.get("hr_fast_bpm")
        motion_z = max(z.get("head_jerk", -math.inf), z.get("head_gyro_peak", -math.inf))

        if self._spike is None and motion_z >= sc.motion_z:
            pre = [hr for (ts, hr) in self._hr_fast_hist if t_ns - ts <= 5e9]
            if pre:
                self._spike = (t_ns, float(np.median(pre)), motion_z)
                self._rise_count = 0
        elif self._spike is not None:
            spike_t, pre_hr, spike_z = self._spike
            rising = hr_fast is not None and hr_fast - pre_hr >= sc.hr_rise_bpm
            self._rise_count = self._rise_count + 1 if rising else 0
            if t_ns - spike_t > sc.window_s * 1e9:
                self._spike = None
            elif self._rise_count >= sc.confirm_n:
                self._startle_until = t_ns + int(sc.hold_s * 1e9)
                rise_s = (t_ns - spike_t) / 1e9
                self._startle_reasons = [
                    f"head jerk (z {spike_z:+.1f})",
                    f"HR {hr_fast - pre_hr:+.0f} bpm within {rise_s:.0f} s",
                ]
                self._spike = None

        if hr_fast is not None:
            self._hr_fast_hist.append((t_ns, hr_fast))
        while self._hr_fast_hist and t_ns - self._hr_fast_hist[0][0] > 10e9:
            self._hr_fast_hist.popleft()

    def _reasons(self, state: State, fv: FeatureVector, z: dict[str, float]) -> list[str]:
        if state in (State.STRESSED, State.CALM):
            weights, sign = self.cfg.arousal_weights, 1 if state is State.STRESSED else -1
        elif state in (State.FOCUSED, State.DISTRACTED):
            weights, sign = self.cfg.attention_weights, 1 if state is State.FOCUSED else -1
        elif state is State.FATIGUED:
            weights, sign = {"perclos": 1.0, "blink_dur_ms": 1.0, "blink_rate_pm": 0.5}, 1
        else:
            return []
        scored = sorted(
            ((sign * w * z[k], k) for k, w in weights.items() if k in z),
            reverse=True,
        )
        out = []
        for contribution, name in scored[:3]:
            if contribution <= 0.1:
                break
            out.append(self._describe(name, fv, z))
        return out

    def _describe(self, name: str, fv: FeatureVector, z: dict[str, float]) -> str:
        value = fv.get(name)
        if name in _REASON_FORMAT and value is not None and self.baseline and name in self.baseline.mean:
            label, fmt, mult = _REASON_FORMAT[name]
            return f"{label} {fmt.format((value - self.baseline.mean[name]) * mult)}"
        return f"{name} (z {z[name]:+.1f})"

    # ---- main -------------------------------------------------------------
    def update(self, fv: FeatureVector, extras: dict[str, float] | None = None) -> StateEstimate:
        """`extras` are extra pseudo-z features, e.g. {"object_dwell_s": ...} from perception."""
        t = fv.t_ns
        if reasons := not_worn_reasons(fv):
            # Nothing physiological to estimate; leave axes and hysteresis untouched.
            self._last_t = t
            return StateEstimate(t, State.NO_SIGNAL, 1.0, reasons, None, None, State.NO_SIGNAL)
        dt_s = 0.0 if self._last_t is None else (t - self._last_t) / 1e9
        self._last_t = t

        z = self._zscores(fv, extras)
        self._arousal = self._smooth(self._arousal, self._axis(z, self.cfg.arousal_weights), dt_s)
        self._attention = self._smooth(self._attention, self._axis(z, self.cfg.attention_weights), dt_s)

        sustained = None
        if self._arousal is not None:
            self._arousal_hist.append((t, self._arousal))
            window = self.cfg.thresholds.stressed_sustain_s * 1e9
            while t - self._arousal_hist[0][0] > window:
                self._arousal_hist.popleft()
            if t - self._arousal_hist[0][0] >= 0.75 * window:
                sustained = float(np.mean([a for _, a in self._arousal_hist]))

        candidate, margin = self._candidate(z, sustained)
        if candidate == self._shown or self._shown is State.UNKNOWN or candidate is State.UNKNOWN:
            self._shown, self._pending = candidate, None
        elif candidate != self._pending:
            self._pending, self._pending_since = candidate, t
        elif t - self._pending_since >= self.cfg.min_state_s * 1e9:
            self._shown, self._pending = candidate, None

        self._update_startle(t, fv, z)
        if t <= self._startle_until:
            state, confidence, reasons = State.STARTLED, 0.9, list(self._startle_reasons)
        else:
            state = self._shown
            confidence = _confidence(margin) if state == candidate else 0.5
            reasons = self._reasons(state, fv, z)

        return StateEstimate(
            t_ns=t,
            state=state,
            confidence=confidence,
            reasons=reasons,
            arousal=self._arousal,
            attention=self._attention,
            candidate=candidate,
            z=z,
        )
