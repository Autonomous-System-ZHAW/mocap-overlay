"""Virtual pinhole camera that covers the whole field of view of the real (distorted) camera.

Renderers (our OpenGL renderer, Blender) only know ideal pinhole cameras. We render with a
pinhole camera that has the real camera's pose and focal length, but a larger image
("overscan") so that every real pixel is inside it, and convert between the two images
with lookup maps:

    distort map:    real pixel   -> virtual pixel   (render -> real lens look, cv2.remap)
    undistort map:  virtual pixel -> real pixel     (real video -> pinhole look, e.g. Blender background)
"""

from dataclasses import dataclass

import cv2
import numpy as np

from calib.camera import Camera
from calib.intrinsics import project_points, undistort_points


@dataclass
class VirtualCamera:
    K: np.ndarray  # (3, 3) pinhole intrinsics of the virtual image
    size: tuple[int, int]  # (width, height) of the virtual image
    map_x: np.ndarray  # (H, W) float32: for each real pixel, x in the virtual image (-1 = invalid)
    map_y: np.ndarray
    invalid_fraction: float  # real pixels without a valid undistortion (edge of the calibrated FOV)

    @classmethod
    def from_camera(cls, cam: Camera, scale: float = 1.0, max_half_angle_deg: float = 80.0) -> "VirtualCamera":
        w, h = cam.image_size
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pix = np.column_stack([u.ravel(), v.ravel()])
        norm = undistort_points(cam.model, pix, cam.K, cam.dist)
        # The iterative undistortion can fail at the edge of the calibrated field of view: keep only
        # pixels that project back onto themselves.
        back = project_points(cam.model, np.column_stack([norm, np.ones(len(norm))]), np.zeros(3), np.zeros(3),
                              cam.K, cam.dist)
        limit = np.tan(np.radians(max_half_angle_deg))
        valid = (np.linalg.norm(back - pix, axis=1) < 1.0) & (np.abs(norm) < limit).all(axis=1)
        lo, hi = norm[valid].min(axis=0), norm[valid].max(axis=0)
        f = float(np.mean([cam.K[0, 0], cam.K[1, 1]])) * scale
        size = (int(np.ceil((hi[0] - lo[0]) * f)) + 2, int(np.ceil((hi[1] - lo[1]) * f)) + 2)
        K = np.array([[f, 0, -lo[0] * f + 1], [0, f, -lo[1] * f + 1], [0, 0, 1]])
        mx, my = norm[:, 0] * f + K[0, 2], norm[:, 1] * f + K[1, 2]
        mx[~valid], my[~valid] = -1, -1
        return cls(K, size, mx.reshape(h, w).astype(np.float32), my.reshape(h, w).astype(np.float32),
                   float(1 - valid.mean()))

    def undistort_maps(self, cam: Camera, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Maps to resample the real image into the virtual pinhole image (optionally scaled down).
        Virtual pixels outside the real image map to -1 (black)."""
        w, h = int(round(self.size[0] * scale)) // 2 * 2, int(round(self.size[1] * scale)) // 2 * 2
        u, v = np.meshgrid((np.arange(w) + 0.5) / scale - 0.5, (np.arange(h) + 0.5) / scale - 0.5)
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = (v - self.K[1, 2]) / self.K[1, 1]
        rays = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
        px = project_points(cam.model, rays, np.zeros(3), np.zeros(3), cam.K, cam.dist)
        mx, my = px[:, 0].reshape(h, w), px[:, 1].reshape(h, w)
        # Only where the forward/backward mapping is consistent (inside the calibrated FOV).
        back_x = cv2.remap(self.map_x, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                           borderValue=-1)
        bad = np.abs(back_x - u) > 2.0
        mx[bad], my[bad] = -1, -1
        return mx.astype(np.float32), my.astype(np.float32), (w, h)
