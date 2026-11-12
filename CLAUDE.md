# CLAUDE.md

## Git

- **Never add yourself as co-author.** Do not add `Co-Authored-By: Claude …` or any other AI attribution lines to commit messages or PR descriptions.
- Commit only milestones (code, tests, docs). Never commit `.plan/`, which holds the private working plan and is gitignored.
- When a milestone is committed, update its row in `.plan/PLAN.md` (status + commit hash).

## Project

- Aria Gen 2 demo: YOLO on the RGB camera, the gaze-attended object, physiological signals, and an estimated wearer state, shown on a web dashboard.
- Plan and milestones are in `.plan/PLAN.md`. Stream availability findings are in `docs/streams.md`.
- Develop on recorded VRS files first. Live streaming comes last.

## Environment

- Use `uv` only (no pip/conda). Python is pinned to 3.12 (`projectaria-tools` needs it).
- Run: `uv run aria-demo <command>`.
- Tests: `uv run pytest`.
- Lint/format: `uv run ruff check . && uv run ruff format .`.

## Conventions

- Code lives in `src/aria_demo/`, organised by pipeline stage (`sources/`, `calibration/`, `perception/`, `physio/`, `state/`, `dashboard/`). Add a package when its milestone starts, not before.
- Import `projectaria_tools` lazily inside functions that need it, so pure logic stays testable without the SDK or device.
- Tests use small synthetic fixtures, never VRS files.
