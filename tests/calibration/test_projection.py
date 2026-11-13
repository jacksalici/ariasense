from types import SimpleNamespace

import numpy as np
import pytest

from ariasense.calibration.projection import (
    GazeProjector,
    gaze_point_cpf,
    needs_upright_rotation,
    rotate_pixel_cw90,
    upright,
)
from ariasense.types import EyeGazeSample

W, H, F = 640, 480, 300.0


def pinhole(p: np.ndarray) -> np.ndarray:
    return np.array([F * p[0] / p[2] + W / 2, F * p[1] / p[2] + H / 2])


def make_projector(rotate: bool = False, T=None) -> GazeProjector:
    return GazeProjector(np.eye(4) if T is None else T, pinhole, (W, H), rotate_upright=rotate)


@pytest.mark.parametrize("yaw,pitch,depth", [(0, 0, 1), (0.3, 0, 1), (0.3, -0.2, 2), (-0.5, 0.4, 0.7)])
def test_gaze_point_matches_sdk(yaw, pitch, depth):
    mps = pytest.importorskip("projectaria_tools.core.mps")
    expected = mps.get_eyegaze_point_at_depth(yaw, pitch, depth).ravel()
    np.testing.assert_allclose(gaze_point_cpf(yaw, pitch, depth), expected, atol=1e-9)


def test_rotate_pixel_matches_rot90():
    raw = np.zeros((H, W), dtype=np.uint8)
    u, v = 100, 30
    raw[v, u] = 255
    rotated = upright(raw)
    ru, rv = rotate_pixel_cw90(u, v, H)
    assert rotated.shape == (W, H)
    assert rotated[int(rv), int(ru)] == 255


def test_straight_gaze_hits_principal_point():
    assert make_projector().project(0.0, 0.0, 1.0) == pytest.approx((W / 2, H / 2))


def test_yaw_moves_along_x_and_depth_default_is_used():
    proj = make_projector()
    u, v = proj.project(np.arctan(0.1), 0.0, None)  # tan(yaw)=0.1 -> x/z = 0.1
    assert u == pytest.approx(W / 2 + F * 0.1)
    assert v == pytest.approx(H / 2)


def test_point_behind_camera_is_none():
    flip = np.diag([1.0, 1.0, -1.0, 1.0])  # camera looks the opposite way
    assert make_projector(T=flip).project(0.0, 0.0, 1.0) is None


def test_rescale_to_delivered_image_and_rotate():
    proj = make_projector(rotate=True)
    # delivered raw image is half the calibration resolution
    u, v = proj.project(0.0, 0.0, 1.0, raw_image_hw=(H // 2, W // 2))
    assert (u, v) == pytest.approx(rotate_pixel_cw90(W / 4, H / 4, H // 2))


def test_invalid_sample_is_not_projected():
    gaze = EyeGazeSample(t_ns=0, yaw=0, pitch=0, depth_m=1, valid=False)
    assert make_projector().project_sample(gaze) is None


def test_from_device_calibration_with_sdk_camera():
    calib_mod = pytest.importorskip("projectaria_tools.core.calibration")
    sophus = pytest.importorskip("projectaria_tools.core.sophus")
    camera = calib_mod.get_linear_camera_calibration(W, H, F, "camera-rgb")
    device = SimpleNamespace(
        get_camera_calib=lambda label: camera,
        get_transform_cpf_sensor=lambda label: sophus.SE3.from_matrix(np.eye(4)),
    )
    proj = GazeProjector.from_device_calibration(device, rotate_upright=False)
    u, v = proj.project(0.0, 0.0, 1.0)
    cx, cy = camera.get_principal_point()
    assert (u, v) == pytest.approx((cx, cy))


@pytest.mark.parametrize("version,expected", [("Gen1", True), ("Gen2", False)])
def test_rotation_only_for_gen1(version, expected):
    device = SimpleNamespace(get_device_version=lambda: SimpleNamespace(name=version))
    assert needs_upright_rotation(device) is expected


def test_from_device_calibration_missing_camera():
    device = SimpleNamespace(
        get_camera_calib=lambda label: None,
        get_transform_cpf_sensor=lambda label: None,
        get_device_version=lambda: SimpleNamespace(name="Gen2"),
    )
    with pytest.raises(ValueError, match="camera-rgb"):
        GazeProjector.from_device_calibration(device)
