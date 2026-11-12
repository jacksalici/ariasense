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


@app.command()
def overlay(
    vrs: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="VRS recording.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output MP4.")] = Path("runs/overlay.mp4"),
    show: Annotated[bool, typer.Option(help="Also show frames in a window (q to quit).")] = False,
    speed: Annotated[float, typer.Option(help="Playback speed; 0 = as fast as possible.")] = 0.0,
    trail: Annotated[float, typer.Option(help="Gaze trail length in seconds.")] = 0.5,
    depth: Annotated[float, typer.Option(help="Gaze depth (m) when the device gives none.")] = 1.0,
) -> None:
    """M1 checkpoint: draw the gaze dot on the upright RGB video."""
    from aria_demo.calibration.projection import GazeProjector
    from aria_demo.overlay import OverlayStats, render, write_video
    from aria_demo.sources.base import Signal
    from aria_demo.sources.vrs_replay import VrsReplaySource

    source = VrsReplaySource(vrs, signals={Signal.RGB, Signal.EYE_GAZE}, speed=speed)
    signals = {signal for signal, _ in source.streams.values()}
    if Signal.EYE_GAZE not in signals:
        typer.secho("Warning: no eye-gaze stream in this recording; output has no gaze.", fg="yellow")
    projector = GazeProjector.from_device_calibration(source.device_calibration(), default_depth_m=depth)

    stats = OverlayStats()
    n = write_video(render(source.samples(), projector, trail_s=trail, stats=stats), out, show=show)
    typer.echo(f"Wrote {n} frames to {out}")
    typer.echo(stats.summary())


if __name__ == "__main__":
    app()
