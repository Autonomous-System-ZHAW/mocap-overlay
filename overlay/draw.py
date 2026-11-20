"""Draw Vicon-world geometry into a camera image (anti-aliased OpenCV primitives)."""

import cv2
import numpy as np

from calib.camera import Camera
from calib.transforms import apply

AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR: x red, y green, z blue


def _subdivide(pts: np.ndarray, step_m: float) -> np.ndarray:
    """Insert points along each segment so straight 3D lines bend correctly with lens distortion."""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def polyline(img, cam: Camera, pts_world, color, thickness: int = 2, step_m: float = 0.05) -> None:
    """3D polyline; parts behind the camera or outside the valid field of view are skipped."""
    px, ok = cam.project(_subdivide(np.asarray(pts_world, float), step_m))
    run = []
    for p, v in zip(px, ok):
        if v:
            run.append(p)
            continue
        if len(run) > 1:
            cv2.polylines(img, [np.round(np.array(run) * 16).astype(np.int32)], False, color, thickness, cv2.LINE_AA, 4)
        run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(np.array(run) * 16).astype(np.int32)], False, color, thickness, cv2.LINE_AA, 4)


def points(img, cam: Camera, pts_world, color, radius: int = 6, thickness: int = 2) -> None:
    px, ok = cam.project(pts_world)
    for p in px[ok]:
        cv2.circle(img, tuple(np.round(p * 16).astype(int)), radius * 16, color, thickness, cv2.LINE_AA, 4)


def axes(img, cam: Camera, T_world_obj: np.ndarray, length: float = 0.2, thickness: int = 3) -> None:
    origin = T_world_obj[:3, 3]
    for i, color in enumerate(AXIS_COLORS):
        polyline(img, cam, [origin, origin + T_world_obj[:3, i] * length], color, thickness)


def floor_grid(img, cam: Camera, x_range, y_range, step: float, color=(200, 200, 200), z: float = 0.0) -> None:
    for x in np.arange(x_range[0], x_range[1] + 1e-9, step):
        polyline(img, cam, [[x, y_range[0], z], [x, y_range[1], z]], color, 1)
    for y in np.arange(y_range[0], y_range[1] + 1e-9, step):
        polyline(img, cam, [[x_range[0], y, z], [x_range[1], y, z]], color, 1)


def board_outline(img, cam: Camera, board: cv2.aruco.CharucoBoard, T_world_board: np.ndarray, color) -> None:
    sx, sy = board.getChessboardSize()
    w, h = sx * board.getSquareLength(), sy * board.getSquareLength()
    rect = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0], [0, 0, 0]], float)
    polyline(img, cam, apply(T_world_board, rect), color, 2, step_m=0.02)
    points(img, cam, apply(T_world_board, board.getChessboardCorners()), color, radius=3, thickness=-1)
