"""Rigid transforms as 4x4 matrices. Naming: T_<target>_<source> maps source -> target."""

import cv2
import numpy as np


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def from_rvec(rvec, tvec) -> np.ndarray:
    return make_T(cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0], tvec)


def to_rvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def average_poses(Ts: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Mean of nearby poses. Returns (T_mean, translation std [m], max rotation deviation [deg])."""
    Ts = np.asarray(Ts)
    t = Ts[:, :3, 3]
    R = project_to_so3(Ts[:, :3, :3].mean(axis=0))
    t_std = float(np.linalg.norm(t.std(axis=0)))
    r_dev = max(rotation_angle_deg(R.T @ Ri) for Ri in Ts[:, :3, :3])
    return make_T(R, t.mean(axis=0)), t_std, r_dev
