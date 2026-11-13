# Aria Gen 2: available streams

This page records which signals we can actually use. It has two parts:

1. **API-level findings**, read from `projectaria-tools 2.3.0` / `projectaria-client-sdk 2.5.1`.
2. **Per-recording reports**, appended by:

   ```bash
   uv run aria-demo inspect data/<session>.vrs -o docs/streams.md
   ```

## 1. API-level findings

| Signal | API | Fields | Implication for the demo |
|---|---|---|---|
| Eye gaze (on-device) | `get_eye_gaze_data_*` → `EyeGaze` | `yaw`, `pitch` (+ low/high bounds), `depth`, `spatial_gaze_point_in_cpf`, `combined_gaze_valid` | Gaze → RGB projection, fixations/saccades |
| Blinks | `EyeGaze.vergence` | `left_blink`, `right_blink` (+ `_valid`), per-eye yaw/pitch | Blink rate, blink duration, PERCLOS |
| **Pupil diameter** | — | **Not exposed** in `EyeGaze` | ⚠️ The arousal feature must come from elsewhere: confirm on a recording; fallback = estimate from eye-tracking camera images, or drop pupil and rely on HR/HRV |
| PPG | `get_ppg_data_*` → `PpgData` | `value`, `led_current_ma`, `integration_time_us` | **Raw signal only**, so HR/HRV must be computed ourselves |
| Ambient light | `get_als_data_*` → `AlsData` | `lux`, per-channel normalised values, UV/IR flux | Luminance correction for any pupil-based feature |
| IMU | `get_imu_data_*` | accel, gyro | Head-motion jerk, PPG quality gate |
| VIO / pose | `get_vio_data_*`, `get_vio_high_freq_data_*` | device pose | Later: 3D gaze map (deferred demo #2) |
| Hand pose | `get_hand_pose_data_*` | — | Not used yet |
| Device calibration | `get_device_calibration()` | CPF ↔ camera transforms | Gaze → RGB pixel |

### Connectivity (from the Gen 2 docs)

- Pairing and starting/stopping a stream need **USB**.
- Data can then flow over USB, Wi-Fi (`--interface wifi_sta`) or the device hotspot.
- Recording works standalone (on-device storage).

### Findings and open questions

- [x] Which of the streams above are present in a Gen 2 recording, and at what rate? → see s01_desk below:
  - RGB: 10 Hz.
  - Eye gaze and hand tracking: 30 Hz.
  - PPG: 128 Hz.
  - ALS: 10 Hz.
  - IMU: 2 × 800 Hz.
- [x] Is any pupil-size signal present? → **No** stream or field carries pupil size.
- Other findings on s01_desk:
  - **The RGB image is already upright**: 2560×1920, FISHEYE624. Unlike Gen 1, no 90° rotation is needed.
  - **Blink flags are inverted**: `vergence.left/right_blink` are True while the eye is open.
  - YOLO tags the wearer's own forearm as `person`. Those boxes are filtered out when they touch the bottom edge.
- [x] Which streams arrive **live**? Checked with the glasses over USB on 2026-11-13 using `aria-demo stream-check`:

  | stream | `mp_streaming_demo` (default) | `profile9` |
  |---|---|---|
  | RGB | **10 Hz**, 2016×1512 | 5 Hz, 2560×1920 |
  | eye gaze | 30 Hz | 30 Hz |
  | PPG | 128 Hz | 128 Hz |
  | IMU (one, decimated ÷4) | 200 Hz | 200 Hz |
  | calibration | yes, rescaled to the stream resolution | yes |
  | ALS | **no callback**, so `lux` is unavailable live | — |

  - PPG and IMU arrive in **batches up to ~1 s behind** the other streams. The reorder buffer now orders each stream on its own, so nothing is dropped.
  - Capture-to-host latency (TicSync) is a median of **110 ms** (p95 118 ms). The first measurement, 310 ms, included a 0.3 s reorder hold, since reduced to 0.1 s.
- [x] Wi-Fi (`wifi_sta`) on "hermes", a **phone hotspot**, 2026-11-13:
  - While the USB cable is plugged in, the glasses stream **over USB** whatever interface is requested (40 MB over USB, 1 MB over Wi-Fi).
  - After unplugging (with `keep_streaming_on_disconnection`), USB traffic stopped and **no data arrived over Wi-Fi**.
  - The laptop can't resolve the glasses' address on hermes (ARP "incomplete", no ping), so the network has **client isolation**: devices on it can't reach each other.
  - Retested with a stream started on the glasses themselves (`WIFI_STA`, glasses at `10.65.138.46`): still ARP "incomplete", and nothing reached the laptop. The phone hotspot itself isolates its clients.
  - Conclusion: `wifi_sta` needs a network that lets devices reach each other, such as your own router or travel router. This phone hotspot doesn't.
- [x] **Wi-Fi works on a normal home network** ("ThePromisedLan", 2026-11-13). Laptop at `192.168.4.166`, glasses at `192.168.4.34`; ping replies in 45–155 ms.
  - With devices able to reach each other, the stream goes over **Wi-Fi even while the cable is in**: 16 MB over Wi-Fi, 0 MB over USB. On the isolated hotspot it had fallen back to USB.
  - **Unplugged and worn:** the stream kept running over Wi-Fi at ~1.3 MB/s. All streams arrived at full rate (RGB 10, gaze 30, PPG 128, IMU 200 Hz) and no new frames were skipped. Gaze, the attended object, heart rate (75 bpm), HRV, blinks and state (Focused) were all live.
  - Latency was **~180–260 ms** over Wi-Fi, against ~110 ms over USB.
- [ ] The glasses' own hotspot (`wifi_sap`), for venues without a suitable network. Not tested yet.

## 2. Per-recording reports


## Recording `s01_desk.vrs`

Device: DeviceVersion.Gen2

| Label | Type | Stream ID | Samples | Nominal Hz | Measured Hz | Duration s |
|---|---|---|---:|---:|---:|---:|
| battery-status | ??? | `502-1` | 0 | 0.0 | 0.0 | 0.0 |
| temperature | ??? | `246-1` | 384 | 1.0 | 25.5 | 15.0 |
| als | ALS | `500-1` | 153 | 10.0 | 10.0 | 15.2 |
| mic | AUDIO | `231-1` | 764 | 16000.0 | 50.0 | 15.3 |
| baro0 | BAROMETER | `247-1` | 764 | 50.0 | 49.9 | 15.3 |
| bluetooth | BLUETOOTH | `283-1` | 10 | 0.0 | 0.0 | 0.0 |
| eyegaze | EYE_GAZE | `373-1` | 458 | 30.0 | 30.0 | 15.2 |
| gps-app | GPS | `281-1` | 13 | 1.0 | 1.0 | 12.0 |
| handtracking | HAND_POSE | `371-1` | 459 | 30.0 | 30.0 | 15.3 |
| camera-et-left | IMAGE | `211-1` | 77 | 5.0 | 5.0 | 15.2 |
| camera-et-right | IMAGE | `211-2` | 77 | 5.0 | 5.0 | 15.2 |
| camera-rgb | IMAGE | `214-1` | 153 | 10.0 | 10.0 | 15.2 |
| slam-front-left | IMAGE | `1201-1` | 459 | 30.0 | 30.0 | 15.3 |
| slam-front-right | IMAGE | `1201-2` | 459 | 30.0 | 30.0 | 15.3 |
| slam-side-left | IMAGE | `1201-3` | 459 | 30.0 | 30.0 | 15.3 |
| slam-side-right | IMAGE | `1201-4` | 459 | 30.0 | 30.0 | 15.3 |
| imu-left | IMU | `1202-1` | 12123 | 800.0 | 793.4 | 15.3 |
| imu-right | IMU | `1202-2` | 12269 | 800.0 | 802.9 | 15.3 |
| mag0 | MAGNETOMETER | `1203-1` | 1540 | 100.0 | 100.7 | 15.3 |
| ppg | PPG | `248-1` | 1959 | 128.0 | 128.2 | 15.3 |
| vio | VIO | `371-2` | 153 | 10.0 | 10.0 | 15.2 |
| vio_high_frequency | VIO_HIGH_FREQ | `371-3` | 12131 | 800.0 | 793.5 | 15.3 |
| wps | WPS | `282-1` | 6 | -1.0 | 0.0 | 0.0 |

### Fields per probed stream

- **als** (ALS): `blue_channel_normalized`, `capture_timestamp_ns`, `cct`, `clear_channel_normalized`, `clear_flux_watt_per_square_meter`, `exposure_time_us`, `gain_blue`, `gain_clear`, `gain_green`, `gain_ir`, `gain_red`, `gain_uv`, `gain_visible`, `green_channel_normalized`, `ir_channel_normalized`, `ir_flux_watt_per_square_meter`, `lux`, `red_channel_normalized`, `uv_channel_normalized`, `uv_flux_watt_per_square_meter`, `visible_channel_normalized`, `visible_flux_watt_per_square_meter`
- **eyegaze** (EYE_GAZE): `combined_gaze_origin_in_cpf`, `combined_gaze_valid`, `depth`, `pitch`, `pitch_high`, `pitch_low`, `session_uid`, `spatial_gaze_point_in_cpf`, `spatial_gaze_point_valid`, `tracking_timestamp`, `vergence`, `yaw`, `yaw_high`, `yaw_low`, `vergence.left_blink`, `vergence.left_blink_valid`, `vergence.left_gaze_valid`, `vergence.left_pitch`, `vergence.left_yaw`, `vergence.left_yaw_high`, `vergence.left_yaw_low`, `vergence.right_blink`, `vergence.right_blink_valid`, `vergence.right_gaze_valid`, `vergence.right_pitch`, `vergence.right_yaw`, `vergence.right_yaw_high`, `vergence.right_yaw_low`, `vergence.tx_left_eye`, `vergence.tx_right_eye`, `vergence.ty_left_eye`, `vergence.ty_right_eye`, `vergence.tz_left_eye`, `vergence.tz_right_eye`
- **handtracking** (HAND_POSE): `left_hand`, `right_hand`, `tracking_timestamp`
- **ppg** (PPG): `capture_timestamp_ns`, `integration_time_us`, `led_current_ma`, `value`
- **vio** (VIO): `angular_velocity_in_bodyimu`, `camera_serials`, `capture_timestamp_ns`, `frame_id`, `frontend_session_uid`, `gravity_in_odometry`, `linear_velocity_in_odometry`, `online_calib`, `pose_quality`, `status`, `transform_bodyimu_device`, `transform_odometry_bodyimu`, `unix_timestamp_ns`, `visual_tracking_quality`
