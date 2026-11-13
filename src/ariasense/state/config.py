"""State-estimator settings, loaded from `configs/state.yaml` (all fields have defaults)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG = Path("configs/state.yaml")


class Thresholds(BaseModel):
    """All in baseline z-units unless the name says otherwise."""

    focused_attention: float = 0.8
    focused_max_arousal: float = 1.5
    stressed_arousal: float = 1.2
    stressed_sustain_s: float = 20.0
    distracted_attention: float = -0.8
    calm_max_arousal: float = 0.5
    fatigue_perclos_z: float = 2.0
    fatigue_blink_dur_z: float = 1.5
    fatigue_max_arousal: float = 0.5


class StartleConfig(BaseModel):
    # On s01_desk, ordinary desk movement reached z 3.2 and the 3-beat HR wobbled ±6 bpm.
    motion_z: float = 4.0  # head jerk or peak rotation spike
    hr_rise_bpm: float = 8.0  # instant HR rise after the spike
    confirm_n: int = 2  # the rise must hold for this many consecutive estimates
    window_s: float = 6.0  # time after the spike to look for the HR rise
    hold_s: float = 5.0  # how long "Startled" stays on screen


class StateConfig(BaseModel):
    arousal_weights: dict[str, float] = Field(
        default_factory=lambda: {"hr_bpm": 1.0, "rmssd_ms": -0.7, "head_gyro_rms": 0.3}
    )
    attention_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fix_dur_ms": 1.0,
            "fix_ratio": 0.7,
            "gaze_disp_deg": -1.0,
            "blink_rate_pm": -0.3,
            "object_dwell_s": 0.5,
        }
    )
    z_clip: float = 4.0
    smoothing_s: float = 5.0  # EMA time constant for the two axes
    min_state_s: float = 3.0  # a new state must hold this long before it is shown
    thresholds: Thresholds = Field(default_factory=Thresholds)
    startle: StartleConfig = Field(default_factory=StartleConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> StateConfig:
        path = path or DEFAULT_CONFIG
        if not path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})
