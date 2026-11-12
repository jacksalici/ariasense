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


if __name__ == "__main__":
    app()
