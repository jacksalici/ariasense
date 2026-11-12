"""Shared test doubles."""

from types import SimpleNamespace

import numpy as np


class FakeCalibration:
    """Gen 2 device calibration with a W x H pinhole RGB camera at the CPF origin."""

    def __init__(self, w: int = 64, h: int = 48):
        self.w, self.h = w, h

    def get_device_version(self):
        return SimpleNamespace(name="Gen2")

    def get_camera_calib(self, label):
        w, h = self.w, self.h
        return SimpleNamespace(
            get_image_size=lambda: (w, h),
            project=lambda p: np.array([[w / 2 + 30 * p[0, 0] / p[2, 0]], [h / 2 + 30 * p[1, 0] / p[2, 0]]]),
        )

    def get_transform_cpf_sensor(self, label):
        return SimpleNamespace(inverse=lambda: SimpleNamespace(to_matrix=lambda: np.eye(4)))
