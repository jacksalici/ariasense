# Aria Gen 2: available streams

This page records which signals we can actually use. It has two parts:

1. **API-level findings**, read from `projectaria-tools 2.3.0` / `projectaria-client-sdk 2.5.1`.
2. **Per-recording reports**, appended by:

   ```bash
   uv run aria-demo inspect data/<session>.vrs -o docs/streams.md
   ```

## 1. API-level findings (no device yet)

| Signal | API | Fields | Implication for the demo |
|---|---|---|---|
| Eye gaze (on-device) | `get_eye_gaze_data_*` → `EyeGaze` | `yaw`, `pitch` (+ low/high bounds), `depth`, `spatial_gaze_point_in_cpf`, `combined_gaze_valid` | Gaze → RGB projection (M1), fixations/saccades (M3) |
| Blinks | `EyeGaze.vergence` | `left_blink`, `right_blink` (+ `_valid`), per-eye yaw/pitch | Blink rate, blink duration, PERCLOS (M3) |
| **Pupil diameter** | — | **Not exposed** in `EyeGaze` | ⚠️ The arousal feature must come from elsewhere: confirm on a recording; fallback = estimate from eye-tracking camera images, or drop pupil and rely on HR/HRV |
| PPG | `get_ppg_data_*` → `PpgData` | `value`, `led_current_ma`, `integration_time_us` | **Raw signal only**, so HR/HRV must be computed ourselves (M3) |
| Ambient light | `get_als_data_*` → `AlsData` | `lux`, per-channel normalised values, UV/IR flux | Luminance correction for any pupil-based feature |
| IMU | `get_imu_data_*` | accel, gyro | Head-motion jerk, PPG quality gate |
| VIO / pose | `get_vio_data_*`, `get_vio_high_freq_data_*` | device pose | Later: 3D gaze map (deferred demo #2) |
| Hand pose | `get_hand_pose_data_*` | — | Not used yet |
| Device calibration | `get_device_calibration()` | CPF ↔ camera transforms | Gaze → RGB pixel |

### Connectivity (from the Gen 2 docs, to be confirmed on the device)

- Pairing and starting/stopping a stream need **USB**.
- Data can then flow over USB, Wi-Fi (`--interface wifi_sta`) or the device hotspot.
- Recording works standalone (on-device storage).

### Still to answer with a real recording

- [ ] Which of the streams above are present in a Gen 2 recording, and at what rate?
- [ ] Is any pupil-size signal present (maybe in a different stream)?
- [ ] Which streams arrive **live** through the Client SDK streaming callbacks vs. only in recordings?
- [ ] End-to-end latency of USB vs. Wi-Fi streaming.

## 2. Per-recording reports

_None yet. Record sessions following [protocol.md](protocol.md), then run `aria-demo inspect`._
