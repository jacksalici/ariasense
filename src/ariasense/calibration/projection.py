"""Project eye gaze (yaw/pitch/depth in the central pupil frame) onto the RGB image.

Pipeline: gaze angles -> 3D point in CPF -> RGB camera frame -> camera model
(fisheye) -> raw pixel -> optional rescale to the delivered image -> optional
90° clockwise rotation (Gen 1 only) so the image is upright.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ariasense.types import EyeGazeSample

DEFAULT_DEPTH_M = 1.0
RGB_LABEL = "camera-rgb"

Pixel = tuple[float, float]


def gaze_point_cpf(yaw: float, pitch: float, depth_m: float) -> np.ndarray:
    """3D gaze point in CPF; matches projectaria_tools `mps.get_eyegaze_point_at_depth`."""
    direction = np.array([np.tan(yaw), np.tan(pitch), 1.0])
    return depth_m * direction / np.linalg.norm(direction)


def rotate_pixel_cw90(u: float, v: float, raw_height: int) -> Pixel:
    """Pixel position after rotating the image 90° clockwise (`np.rot90(img, k=-1)`)."""
    return raw_height - 1 - v, u


def upright(image: np.ndarray) -> np.ndarray:
    """Rotate a raw Gen 1 Aria RGB image 90° clockwise so it is upright."""
    return np.ascontiguousarray(np.rot90(image, k=-1))


def needs_upright_rotation(device_calibration: Any) -> bool:
    """Gen 1 RGB frames are stored sideways; Gen 2 frames are already upright (verified on s01_desk)."""
    version = getattr(device_calibration.get_device_version(), "name", "")
    return version == "Gen1"


class GazeProjector:
    def __init__(
        self,
        T_camera_cpf: np.ndarray,
        project: Callable[[np.ndarray], np.ndarray | None],
        calib_size_wh: tuple[int, int],
        rotate_upright: bool = True,
        default_depth_m: float = DEFAULT_DEPTH_M,
    ) -> None:
        self.T_camera_cpf = np.asarray(T_camera_cpf, dtype=float)
        self._project = project
        self.calib_w, self.calib_h = calib_size_wh
        self.rotate_upright = rotate_upright
        self.default_depth_m = default_depth_m

    @classmethod
    def from_device_calibration(
        cls,
        device_calibration: Any,
        camera_label: str = RGB_LABEL,
        rotate_upright: bool | None = None,
        default_depth_m: float = DEFAULT_DEPTH_M,
    ) -> GazeProjector:
        """`rotate_upright=None` decides from the device: only Gen 1 RGB images are sideways."""
        if rotate_upright is None:
            rotate_upright = needs_upright_rotation(device_calibration)
        camera = device_calibration.get_camera_calib(camera_label)
        T_cpf_camera = device_calibration.get_transform_cpf_sensor(camera_label)
        if camera is None or T_cpf_camera is None:
            raise ValueError(f"No calibration for camera '{camera_label}'")
        w, h = (int(x) for x in camera.get_image_size())
        return cls(
            T_camera_cpf=T_cpf_camera.inverse().to_matrix(),
            project=lambda p: camera.project(p.reshape(3, 1)),
            calib_size_wh=(w, h),
            rotate_upright=rotate_upright,
            default_depth_m=default_depth_m,
        )

    def project(
        self,
        yaw: float,
        pitch: float,
        depth_m: float | None = None,
        raw_image_hw: tuple[int, int] | None = None,
    ) -> Pixel | None:
        """Pixel of the gaze point, or None if it falls outside the camera's view.

        `raw_image_hw` is the (height, width) of the delivered raw image when it
        differs from the calibration resolution.
        """
        depth = depth_m if depth_m is not None and depth_m > 0 else self.default_depth_m
        p_cpf = np.append(gaze_point_cpf(yaw, pitch, depth), 1.0)
        p_cam = (self.T_camera_cpf @ p_cpf)[:3]
        if p_cam[2] <= 0:
            return None
        uv = self._project(p_cam)
        if uv is None:
            return None
        u, v = (float(x) for x in np.ravel(uv))

        raw_h = self.calib_h
        if raw_image_hw is not None:
            raw_h, raw_w = raw_image_hw
            u *= raw_w / self.calib_w
            v *= raw_h / self.calib_h
        return rotate_pixel_cw90(u, v, raw_h) if self.rotate_upright else (u, v)

    def project_sample(
        self, gaze: EyeGazeSample, raw_image_hw: tuple[int, int] | None = None
    ) -> Pixel | None:
        if not gaze.valid:
            return None
        return self.project(gaze.yaw, gaze.pitch, gaze.depth_m, raw_image_hw)
