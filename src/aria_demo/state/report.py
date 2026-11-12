"""Write feature time series to CSV and a multi-panel plot against the baseline band."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aria_demo.state.baseline import Baseline
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
