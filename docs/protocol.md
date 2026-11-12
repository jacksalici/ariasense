# Recording protocol

Save recordings to `data/` (gitignored), named `sNN_<short-name>.vrs`.

## Device setup (once per laptop)

Pairing needs USB.

```bash
uv run aria_gen2 auth pair
```

```bash
uv run aria_gen2 auth check
```

Record, then download:

```bash
uv run aria_gen2 recording start
```

```bash
uv run aria_gen2 recording stop
```

```bash
uv run aria_gen2 recording list
```

```bash
uv run aria_gen2 recording download --help
```

Downloaded files are named `<uuid>.vrs`. Rename them to `sNN_<short-name>.vrs`. Run `--help` on any subcommand to see its options.

## M0: stream check

**s01_desk (2 min).** Seated at a desk with 6–8 everyday objects (cup, phone, book, scissors, keyboard, bottle…). Look at each one for about 5 s, in a known order. This is used for gaze → RGB verification and YOLO.

## M4: state validation

**s02_states (about 8 min).** One continuous recording with the segments below. Note the start time of each segment, or say its name out loud so it can be found in the audio.

| # | Segment | Duration | Expected state |
|---|---|---|---|
| 1 | Sit still, eyes open, neutral | 60 s | Calm (baseline) |
| 2 | Read a dense text paragraph | 90 s | Focused |
| 3 | 2-back task or timed mental arithmetic | 90 s | Stressed |
| 4 | Rest | 30 s | Calm |
| 5 | Unexpected loud sound / jump-scare clip (operator-triggered) | 20 s | Startled |
| 6 | Scan a cluttered room quickly | 60 s | Distracted |

Keep the lighting constant. Stay seated for segments 1–5 to limit PPG motion artefacts.
