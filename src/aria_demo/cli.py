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


@app.command()
def dashboard(
    vrs: VrsArg,
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8000,
    host: Annotated[str, typer.Option(help="Bind address (127.0.0.1 = this machine only).")] = "127.0.0.1",
    speed: Annotated[float, typer.Option(help="Replay speed (1 = real time).")] = 1.0,
    loop: Annotated[bool, typer.Option(help="Restart the replay when it ends.")] = True,
    baseline_s: Annotated[float, typer.Option("--baseline", help="Baseline duration (s).")] = 60.0,
    model: Annotated[str, typer.Option(help="YOLO weights.")] = "yolo26s.pt",
    imgsz: Annotated[int, typer.Option(help="YOLO inference size.")] = 640,
    conf: Annotated[float, typer.Option(help="Detection confidence threshold.")] = 0.35,
    config: Annotated[Path, typer.Option(help="State config YAML.")] = Path("configs/state.yaml"),
    depth: DepthOpt = 1.0,
) -> None:
    """M5: web dashboard replaying a recording (video + gaze + YOLO, signals, estimated state)."""
    from aria_demo.pipeline import vrs_source_factory

    _serve(
        vrs_source_factory(vrs, speed),
        {"source": vrs.name, "mode": "replay"},
        loop=loop,
        port=port,
        host=host,
        baseline_s=baseline_s,
        model=model,
        imgsz=imgsz,
        conf=conf,
        config=config,
        depth=depth,
    )


def _serve(
    source_factory, info: dict, *, loop, port, host, baseline_s, model, imgsz, conf, config, depth
) -> None:
    import uvicorn

    from aria_demo.dashboard.server import create_app
    from aria_demo.perception.detector import YoloDetector
    from aria_demo.pipeline import DemoPipeline
    from aria_demo.state.config import StateConfig

    detector = YoloDetector(model=model, conf=conf, imgsz=imgsz)
    pipeline = DemoPipeline(
        source_factory,
        detector,
        config=StateConfig.load(config),
        baseline_s=baseline_s,
        loop=loop,
        depth_m=depth,
    )
    web = create_app(pipeline, info={**info, "model": model})
    typer.echo(f"Dashboard: http://{host}:{port}  (YOLO {model} on {detector.device})")
    uvicorn.run(web, host=host, port=port, log_level="warning")


InterfaceOpt = Annotated[
    str,
    typer.Option(
        help="usb, wifi_sta (glasses on your Wi-Fi) or wifi_sap (laptop joins the glasses' hotspot)."
    ),
]
ProfileOpt = Annotated[
    str, typer.Option(help="Streaming profile: mp_streaming_demo (RGB 10 Hz) or profile9 (RGB 5 Hz).")
]
SerialOpt = Annotated[str | None, typer.Option(help="Device serial (when several are plugged in).")]
ControlOpt = Annotated[
    bool,
    typer.Option(help="Connect over USB and start/stop streaming. --no-control only listens."),
]
ReplayOpt = Annotated[
    Path | None,
    typer.Option(
        exists=True, dir_okay=False, help="Feed a VRS through the SDK streaming path instead (no device)."
    ),
]


def _live_factory(
    replay: Path | None, interface: str, profile: str, serial: str | None, control: bool, record
):
    from aria_demo.pipeline import live_source_factory, sdk_replay_source_factory

    if replay is not None:
        return sdk_replay_source_factory(replay), {
            "source": f"{replay.name} via SDK stream",
            "mode": "sdk-replay",
        }
    factory = live_source_factory(
        interface=interface, profile=profile, serial=serial, control_device=control, record_to=record
    )
    return factory, {"source": f"glasses · {interface} · {profile}", "mode": "live"}


@app.command()
def live(
    interface: InterfaceOpt = "usb",
    profile: ProfileOpt = "mp_streaming_demo",
    serial: SerialOpt = None,
    control: ControlOpt = True,
    record: Annotated[Path | None, typer.Option(help="Also save the stream to this VRS file.")] = None,
    replay: ReplayOpt = None,
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8000,
    host: Annotated[str, typer.Option(help="Bind address (127.0.0.1 = this machine only).")] = "127.0.0.1",
    baseline_s: Annotated[float, typer.Option("--baseline", help="Baseline duration (s).")] = 60.0,
    model: Annotated[str, typer.Option(help="YOLO weights.")] = "yolo26s.pt",
    imgsz: Annotated[int, typer.Option(help="YOLO inference size.")] = 640,
    conf: Annotated[float, typer.Option(help="Detection confidence threshold.")] = 0.35,
    config: Annotated[Path, typer.Option(help="State config YAML.")] = Path("configs/state.yaml"),
    depth: DepthOpt = 1.0,
) -> None:
    """M6: the dashboard on the live stream from the glasses."""
    factory, info = _live_factory(replay, interface, profile, serial, control, record)
    _serve(
        factory,
        info,
        loop=False,
        port=port,
        host=host,
        baseline_s=baseline_s,
        model=model,
        imgsz=imgsz,
        conf=conf,
        config=config,
        depth=depth,
    )


@app.command("stream-check")
def stream_check(
    seconds: Annotated[float, typer.Option(help="How long to stream.")] = 15.0,
    interface: InterfaceOpt = "usb",
    profile: ProfileOpt = "mp_streaming_demo",
    serial: SerialOpt = None,
    control: ControlOpt = True,
    replay: ReplayOpt = None,
) -> None:
    """M6: which streams arrive live, at what rate, and with what latency (no YOLO, no UI)."""
    import time

    from aria_demo.pipeline import EXPECTED_STREAMS
    from aria_demo.types import RgbFrame

    factory, info = _live_factory(replay, interface, profile, serial, control, None)
    source = factory()
    typer.echo(f"Streaming from {info['source']} for {seconds:.0f} s…")
    latencies: list[float] = []
    try:
        calib = source.device_calibration()
        cam = calib.get_camera_calib("camera-rgb")
        typer.echo(f"Calibration received (RGB {tuple(int(x) for x in cam.get_image_size())})")
        deadline = time.monotonic() + seconds
        for sample in source.samples():
            if isinstance(sample, RgbFrame) and (ms := source.latency_ms(sample.t_ns)) is not None:
                latencies.append(ms)
            if time.monotonic() >= deadline:
                break
    finally:
        source.close()

    stats = source.stats()
    typer.echo(f"\n{'stream':<10}{'samples':>9}{'rate Hz':>10}")
    for name in sorted({*EXPECTED_STREAMS, *stats.counts}):
        n = stats.counts.get(name, 0)
        flag = "" if n else "   <- missing"
        typer.echo(f"{name:<10}{n:>9}{stats.rates_hz.get(name, 0):>10.1f}{flag}")
    typer.echo(
        f"RGB frames dropped (processing behind): {stats.dropped_rgb}; "
        f"late sensor samples: {stats.late_dropped}"
    )
    if latencies:
        lat = sorted(latencies)
        typer.echo(
            f"Capture-to-host latency: median {lat[len(lat) // 2]:.0f} ms, "
            f"p95 {lat[int(len(lat) * 0.95)]:.0f} ms"
        )
    else:
        typer.echo("Latency: not available (TicSync not valid, or replay mode).")


if __name__ == "__main__":
    app()
