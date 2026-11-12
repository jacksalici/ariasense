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


if __name__ == "__main__":
    app()
