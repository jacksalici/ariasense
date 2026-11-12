"""Write feature time series to CSV and a multi-panel plot against the baseline band."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aria_demo.state.baseline import Baseline
from aria_demo.state.estimator import StateEstimate
from aria_demo.state.evaluate import Segment
from aria_demo.state.features import FEATURES, FeatureVector

PLOTTED = [
    "hr_bpm",
    "rmssd_ms",
    "blink_rate_pm",
    "fix_dur_ms",
    "gaze_disp_deg",
    "head_gyro_rms",
    "head_jerk",
    "lux",
]


def write_csv(rows: Sequence[FeatureVector], baseline: Baseline | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = rows[0].t_ns if rows else 0
    names = list(FEATURES)
    header = ["t_s", *names] + ([f"z_{n}" for n in names] if baseline else [])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fv in rows:
            z = baseline.z(fv) if baseline else {}
            row = [f"{(fv.t_ns - t0) / 1e9:.2f}"] + [_fmt(fv.get(n)) for n in names]
            if baseline:
                row += [_fmt(z.get(n)) for n in names]
            w.writerow(row)


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4g}"


def plot(
    rows: Sequence[FeatureVector], baseline: Baseline | None, path: Path, baseline_end_s: float | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = rows[0].t_ns if rows else 0
    t = np.array([(fv.t_ns - t0) / 1e9 for fv in rows])
    fig, axes = plt.subplots(len(PLOTTED), 1, figsize=(11, 2.0 * len(PLOTTED)), sharex=True)
    for ax, name in zip(axes, PLOTTED, strict=True):
        y = np.array([np.nan if (v := fv.get(name)) is None else v for fv in rows], dtype=float)
        ax.plot(t, y, lw=1.4, color="#2563eb")
        if baseline and name in baseline.mean:
            m, s = baseline.mean[name], baseline.std[name]
            ax.axhspan(m - s, m + s, color="#94a3b8", alpha=0.25, lw=0)
            ax.axhline(m, color="#64748b", lw=0.8, ls="--")
        if baseline_end_s:
            ax.axvline(baseline_end_s, color="#f59e0b", lw=0.8)
        ax.set_ylabel(FEATURES[name], fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    title = "Features vs baseline (grey band = mean ± std"
    title += f", baseline = first {baseline_end_s:.0f} s)" if baseline_end_s else ")"
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


STATE_COLORS = {
    "Calibrating": "#e5e7eb",
    "No signal": "#9ca3af",
    "Neutral": "#cbd5e1",
    "Calm": "#22c55e",
    "Focused": "#3b82f6",
    "Distracted": "#f59e0b",
    "Stressed": "#ef4444",
    "Fatigued": "#8b5cf6",
    "Startled": "#111827",
}


def write_states_csv(estimates: Sequence[StateEstimate], t0_ns: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "state", "confidence", "arousal", "attention", "candidate", "reasons"])
        for e in estimates:
            w.writerow(
                [
                    f"{(e.t_ns - t0_ns) / 1e9:.2f}",
                    e.state.value,
                    f"{e.confidence:.2f}",
                    _fmt(e.arousal),
                    _fmt(e.attention),
                    e.candidate.value,
                    "; ".join(e.reasons),
                ]
            )


def _ribbon(ax, times: np.ndarray, names: Sequence[str], label: str) -> None:
    for i, name in enumerate(names):
        t_end = times[i + 1] if i + 1 < len(times) else times[i] + 1
        ax.axvspan(times[i], t_end, color=STATE_COLORS.get(name, "#9ca3af"), lw=0)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)


def plot_states(
    estimates: Sequence[StateEstimate],
    t0_ns: int,
    path: Path,
    labels: Sequence[Segment] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([(e.t_ns - t0_ns) / 1e9 for e in estimates])
    rows = 3 if labels else 2
    fig, axes = plt.subplots(
        rows, 1, figsize=(11, 4.5), sharex=True, gridspec_kw={"height_ratios": [1] * (rows - 1) + [3]}
    )
    _ribbon(axes[0], t, [e.state.value for e in estimates], "estimated")
    if labels:
        ax = axes[1]
        for seg in labels:
            ax.axvspan(seg.start_s, seg.end_s, color=STATE_COLORS[seg.label.value], lw=0)
        ax.set_yticks([])
        ax.set_ylabel("labelled", rotation=0, ha="right", va="center", fontsize=9)

    ax = axes[-1]
    for attr, color in (("arousal", "#ef4444"), ("attention", "#3b82f6")):
        y = np.array([np.nan if getattr(e, attr) is None else getattr(e, attr) for e in estimates])
        ax.plot(t, y, color=color, lw=1.6, label=attr)
    ax.axhline(0, color="#64748b", lw=0.8, ls="--")
    ax.set_ylabel("z (vs baseline)")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    present = list(dict.fromkeys([e.state.value for e in estimates] + [s.label.value for s in labels or []]))
    handles = [Patch(color=STATE_COLORS[n], label=n) for n in present]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=90)
    plt.close(fig)
