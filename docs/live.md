# Live demo with the glasses

## One-time setup

Plug the glasses in over USB, pair them with this laptop, and check:

```bash
uv run aria_gen2 auth pair
```

```bash
uv run aria_gen2 auth check
```

## 1. Check the streams (no YOLO, no UI)

```bash
uv run aria-demo stream-check --interface usb --seconds 15
```

This prints, for each stream (rgb, eye_gaze, ppg, imu), the sample count and rate. It also prints how many RGB frames were skipped and the capture-to-host latency, measured with TicSync.

A stream marked `<- missing` is not in the streaming profile. Try another profile with `--profile`. Ambient light (ALS) has no streaming callback, so the `lux` feature is unavailable live.

## 2. Run the dashboard live

Over USB:

```bash
uv run aria-demo live --interface usb --baseline 60
```

Then open http://127.0.0.1:8000.

- The glasses stream to a receiver on port 6768. Closing the server (Ctrl-C) stops the stream.
- `--record data/s02_states.vrs` also saves the stream, which is a convenient way to record the protocol sessions.
- `--no-control` only listens: start the stream yourself with `aria_gen2 streaming start`.

## Wireless

USB is still needed to **start and stop** streaming. Once the stream is running, you can unplug the cable.

**Your Wi-Fi.** A 5 GHz Wi-Fi 6 router with no firewall in the way is recommended. Connect the glasses to the network, then start streaming over it:

```bash
uv run aria_gen2 device wifi connect --ssid <name> --password <password>
```

```bash
uv run aria-demo live --interface wifi_sta
```

**The glasses' hotspot.** This needs no router, which makes it the safest choice at a venue. Turn on the hotspot (check `aria_gen2 device --help` for the exact command), connect the laptop to it, then run:

```bash
uv run aria-demo live --interface wifi_sap
```

Wi-Fi sends data in batches, so expect more latency than over USB. The dashboard shows it in the top bar: green below 400 ms, amber below 1 s, red above.

## Test the live code path without the glasses

```bash
uv run aria-demo live --replay data/s01_desk.vrs --baseline 15
```

This feeds a recording through the same SDK streaming callbacks used live (`create_replay_source`). Callback wiring, re-ordering, frame dropping and stream health are all exercised; only the network part is skipped.

## Dashboard warnings

- **Not receiving: ppg (…)**: that stream is missing from the profile.
- **No data from the glasses for N s**: the connection has dropped. Check the cable or Wi-Fi.
- **frames skipped** (in the header): processing (decode, YOLO, encode) fell behind and the newest frame was kept. To lower the load, pass `--imgsz 480`, or use a smaller model with `--model yolo26n.pt`.
