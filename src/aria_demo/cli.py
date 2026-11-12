"""Command-line entry point: `aria-demo <command>`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Aria Gen 2 demo: gaze + YOLO + estimated state.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Aria Gen 2 demo tools."""


@app.command()
def inspect(
    vrs: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, help="VRS recording(s).")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Append the report to this Markdown file.")
    ] = None,
) -> None:
    """List the streams, rates and fields in VRS recordings."""
    from aria_demo.inspect_vrs import inspect_vrs

    for path in vrs:
        report = inspect_vrs(path)
        if out is None:
            typer.echo(report)
        else:
            with out.open("a") as f:
                f.write("\n" + report)
            typer.echo(f"Appended report for {path.name} to {out}")


VrsArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="VRS recording.")]
ShowOpt = Annotated[bool, typer.Option(help="Also show frames in a window (q to quit).")]
SpeedOpt = Annotated[float, typer.Option(help="Playback speed; 0 = as fast as possible.")]
TrailOpt = Annotated[float, typer.Option(help="Gaze trail length in seconds.")]
DepthOpt = Annotated[float, typer.Option(help="Gaze depth (m) when the device gives none.")]
ScaleOpt = Annotated[float, typer.Option(help="Resize factor for the output video.")]


def _open_gaze_source(vrs: Path, speed: float, depth: float):
    from aria_demo.calibration.projection import GazeProjector
    from aria_demo.sources.base import Signal
    from aria_demo.sources.vrs_replay import VrsReplaySource

    source = VrsReplaySource(vrs, signals={Signal.RGB, Signal.EYE_GAZE}, speed=speed)
    if Signal.EYE_GAZE not in {signal for signal, _ in source.streams.values()}:
        typer.secho("Warning: no eye-gaze stream in this recording; output has no gaze.", fg="yellow")
    projector = GazeProjector.from_device_calibration(source.device_calibration(), default_depth_m=depth)
    return source, projector


@app.command()
def overlay(
    vrs: VrsArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output MP4.")] = Path("runs/overlay.mp4"),
    show: ShowOpt = False,
    speed: SpeedOpt = 0.0,
    trail: TrailOpt = 0.5,
    depth: DepthOpt = 1.0,
    scale: ScaleOpt = 0.5,
) -> None:
    """M1 checkpoint: draw the gaze dot on the RGB video."""
    from aria_demo.overlay import OverlayStats, render, write_video

    source, projector = _open_gaze_source(vrs, speed, depth)
    stats = OverlayStats()
    frames = render(source.samples(), projector, trail_s=trail, stats=stats)
    n = write_video(frames, out, show=show, scale=scale)
    typer.echo(f"Wrote {n} frames to {out}")
    typer.echo(stats.summary())


@app.command()
def detect(
    vrs: VrsArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output MP4.")] = Path("runs/detect.mp4"),
    model: Annotated[
        str, typer.Option(help="YOLO weights (downloaded to models/ if missing).")
    ] = "yolo26s.pt",
    conf: Annotated[float, typer.Option(help="Detection confidence threshold.")] = 0.35,
    imgsz: Annotated[int, typer.Option(help="YOLO inference size.")] = 960,
    show: ShowOpt = False,
    speed: SpeedOpt = 0.0,
    trail: TrailOpt = 0.5,
    depth: DepthOpt = 1.0,
    scale: ScaleOpt = 0.5,
) -> None:
    """M2 checkpoint: YOLO detections + the object you are looking at, with dwell time."""
    from aria_demo.overlay import OverlayStats, gaze_frames, write_video
    from aria_demo.perception.detector import YoloDetector
    from aria_demo.perception.pipeline import draw, perceive

    source, projector = _open_gaze_source(vrs, speed, depth)
    detector = YoloDetector(model=model, conf=conf, imgsz=imgsz)
    typer.echo(f"YOLO {model} on {detector.device}")

    stats = OverlayStats()
    last = None

    def frames():
        nonlocal last
        for pf in perceive(gaze_frames(source.samples(), projector, trail, stats), detector):
            last = pf
            yield pf.frame.t_ns, draw(pf)

    n = write_video(frames(), out, show=show, scale=scale)
    typer.echo(f"Wrote {n} frames to {out}")
    typer.echo(stats.summary())
    if last is not None:
        typer.echo(f"Attention switches: {last.attention.switches}")
        typer.echo("Time looking at each class:")
        for name, secs in sorted(last.attention.totals_s.items(), key=lambda kv: -kv[1]):
            typer.echo(f"  {name:<16} {secs:6.1f}s")


def _features_and_baseline(vrs: Path, baseline_s: float, step: float):
    """Feature vectors for the whole recording + a baseline from its first `baseline_s`."""
    from aria_demo.sources.base import Signal
    from aria_demo.sources.vrs_replay import VrsReplaySource
    from aria_demo.state.baseline import Baseline
    from aria_demo.state.features import FeatureExtractor, extract

    signals = {Signal.PPG, Signal.EYE_GAZE, Signal.IMU, Signal.ALS}
    source = VrsReplaySource(vrs, signals=signals, speed=0)
    rows = list(extract(source.samples(), FeatureExtractor(step_s=step)))
    if not rows:
        typer.secho("No feature vectors produced.", fg="red")
        raise typer.Exit(1)

    duration = (rows[-1].t_ns - rows[0].t_ns) / 1e9
    if duration < baseline_s * 1.5:
        baseline_s = duration / 2
        typer.secho(
            f"Recording is {duration:.0f} s: using the first {baseline_s:.0f} s as baseline.", fg="yellow"
        )
    baseline = Baseline.from_vectors(fv for fv in rows if (fv.t_ns - rows[0].t_ns) / 1e9 <= baseline_s)
    return rows, baseline, baseline_s


@app.command()
def physio(
    vrs: VrsArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output CSV.")] = Path("runs/features.csv"),
    plot_path: Annotated[
        Path | None, typer.Option("--plot", help="Output PNG (default: next to CSV).")
    ] = None,
    baseline_s: Annotated[float, typer.Option("--baseline", help="Baseline duration (s).")] = 60.0,
    step: Annotated[float, typer.Option(help="Seconds between feature vectors.")] = 1.0,
) -> None:
    """M3 checkpoint: heart rate, HRV, blinks, fixations, head motion -> CSV + plot."""
    from aria_demo.state import report
    from aria_demo.state.features import FEATURES

    rows, baseline, baseline_s = _features_and_baseline(vrs, baseline_s, step)
    report.write_csv(rows, baseline, out)
    plot_path = plot_path or out.with_suffix(".png")
    report.plot(rows, baseline, plot_path, baseline_end_s=baseline_s)
    typer.echo(f"Wrote {len(rows)} feature vectors to {out} and {plot_path}")

    typer.echo(f"\n{'feature':<16}{'baseline':>12}{'rest mean':>12}{'z':>8}")
    rest = [fv for fv in rows if (fv.t_ns - rows[0].t_ns) / 1e9 > baseline_s]
    for name in FEATURES:
        vals = [v for fv in rest if (v := fv.get(name)) is not None]
        if name not in baseline.mean or not vals:
            typer.echo(f"{name:<16}{'-':>12}{'-':>12}{'-':>8}")
            continue
        m = float(sum(vals) / len(vals))
        z = (m - baseline.mean[name]) / baseline.std[name]
        typer.echo(f"{name:<16}{baseline.mean[name]:>12.3g}{m:>12.3g}{z:>8.2f}")


@app.command()
def states(
    vrs: VrsArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output CSV timeline.")] = Path("runs/states.csv"),
    labels: Annotated[
        Path | None, typer.Option(exists=True, help="Labelled segments CSV: start_s,end_s,label.")
    ] = None,
    config: Annotated[Path, typer.Option(help="State config YAML.")] = Path("configs/state.yaml"),
    baseline_s: Annotated[float, typer.Option("--baseline", help="Baseline duration (s).")] = 60.0,
    step: Annotated[float, typer.Option(help="Seconds between estimates.")] = 1.0,
) -> None:
    """M4: estimated state per second (+ evaluation against labelled segments)."""
    from aria_demo.state import report
    from aria_demo.state.config import StateConfig
    from aria_demo.state.estimator import StateEstimator
    from aria_demo.state.evaluate import evaluate, load_labels

    rows, baseline, baseline_s = _features_and_baseline(vrs, baseline_s, step)
    estimator = StateEstimator(StateConfig.load(config), baseline)
    t0 = rows[0].t_ns
    estimates = [estimator.update(fv) for fv in rows if (fv.t_ns - t0) / 1e9 > baseline_s]
    if not estimates:
        typer.secho("Nothing left after the baseline.", fg="red")
        raise typer.Exit(1)

    segments = load_labels(labels) if labels else None
    report.write_states_csv(estimates, t0, out)
    plot_path = out.with_suffix(".png")
    report.plot_states(estimates, t0, plot_path, segments)
    typer.echo(f"Wrote {len(estimates)} estimates to {out} and {plot_path}\n")

    counts: dict[str, int] = {}
    for e in estimates:
        counts[e.state.value] = counts.get(e.state.value, 0) + 1
    typer.echo(
        "Time per state: " + ", ".join(f"{k} {v}s" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    )
    previous = None
    for e in estimates:
        if e.state != previous:
            reasons = "; ".join(e.reasons) or "-"
            typer.echo(
                f"  t={(e.t_ns - t0) / 1e9:6.1f}s  {e.state.value:<11} ({e.confidence:.0%})  {reasons}"
            )
            previous = e.state

    if segments:
        typer.echo("\n" + evaluate(estimates, segments, t0).table())


if __name__ == "__main__":
    app()
