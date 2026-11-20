"""Calibrated camera: intrinsics + pose in the Vicon world. Projects world points to pixels."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from calib.intrinsics import project_points, undistort_points
from calib.io import load_yaml
from calib.transforms import apply, inv_T, to_rvec


@dataclass
class Camera:
    model: str
    image_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray
    T_cam_world: np.ndarray

    @classmethod
    def from_extrinsics_yaml(cls, path: str | Path) -> "Camera":
        d = load_yaml(path)
        i = d["intrinsics"]
        return cls(i["model"], tuple(i["image_size"]), np.array(i["K"], float), np.array(i["dist"], float),
                   np.array(d["T_cam_world"], float))

    @property
    def T_world_cam(self) -> np.ndarray:
        return inv_T(self.T_cam_world)

    @cached_property
    def max_norm_radius(self) -> float:
        """Largest normalised image radius inside the image, plus 10 % margin.

        Distortion polynomials are only valid inside the calibrated field of view; points
        further out can fold back into the image, so they must be discarded.
        """
        w, h = self.image_size
        border = np.array([[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]], float)
        return 1.1 * float(np.linalg.norm(undistort_points(self.model, border, self.K, self.dist), axis=1).max())

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) world points -> (N, 2) pixels and (N,) mask of points that project validly."""
        pts_world = np.asarray(pts_world, float).reshape(-1, 3)
        pc = apply(self.T_cam_world, pts_world)
        z = pc[:, 2]
        valid = z > 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.linalg.norm(pc[:, :2] / z[:, None], axis=1)
        valid &= r < self.max_norm_radius
        px = np.full((len(pts_world), 2), np.nan)
        if valid.any():
            rvec, tvec = to_rvec(self.T_cam_world)
            px[valid] = project_points(self.model, pts_world[valid], rvec, tvec, self.K, self.dist)
        return px, valid
