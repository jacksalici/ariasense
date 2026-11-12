# Live demo with the glasses

## One-time setup

Plug the glasses in over USB, pair them with this laptop, and check:

```bash
uv run aria_gen2 auth pair
```

```bash
uv run aria_gen2 auth check
```

Notes from the first live session:
- **`adb` is not needed** if you don't pass `--serial`. The default connection works over USB networking (the glasses show up as `10.37.25.x`). Passing `--serial` requires `adb` on the PATH.
- **Streaming certificates** are created in `~/.aria/streaming-certs/persistent` the first time streaming starts. The device must start streaming *before* the receiver loads them; `LiveSource` already does this.
- If a run crashes and leaves the glasses streaming, the next start fails with "User session already started". Clear it with:

```bash
uv run aria_gen2 streaming stop
```

## 1. Check the streams (no YOLO, no UI)

```bash
uv run aria-demo stream-check --interface usb --seconds 15
```

This prints, for each stream (rgb, eye_gaze, ppg, imu), the sample count and rate. It also prints how many RGB frames were skipped and the capture-to-host latency, measured with TicSync.

A stream marked `<- missing` is not in the streaming profile. Try another profile with `--profile`. `mp_streaming_demo` is the default (RGB at 10 Hz); `profile9` gives RGB at only 5 Hz. Ambient light (ALS) has no streaming callback, so the `lux` feature is unavailable live.

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

**First check that the two devices can reach each other.** Many university and enterprise networks (e.g. "hermes") isolate clients, so the stream never arrives. With the laptop on the same network as the glasses, run:

```bash
ping -c 3 <glasses Wi-Fi IP>
```

The glasses' Wi-Fi IP is shown by `uv run aria_gen2 device status`. If ping fails and `arp -an` lists the glasses as `(incomplete)`, the network isolates clients. Use the glasses' hotspot or your own router instead.

While the USB cable is plugged in, the glasses stream over USB even with `--interface wifi_*`. Wi-Fi only carries the stream after you unplug. `aria-demo live` sets `keep_streaming_on_disconnection` for Wi-Fi interfaces, so the glasses keep streaming once unplugged. They also **keep streaming after you stop the dashboard**, until you plug them back in and run `uv run aria_gen2 streaming stop`, so mind the battery.


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
