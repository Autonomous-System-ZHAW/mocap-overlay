"""Load a recording made by record/recorder.py and query Vicon poses at arbitrary times."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml


@dataclass
class PoseTrack:
    """Time series of one Vicon subject. Occluded samples have NaN position / quaternion."""

    t: np.ndarray  # (N,) host time [s]
    pos: np.ndarray  # (N, 3) [m]
    quat: np.ndarray  # (N, 4) x, y, z, w
    occluded: np.ndarray  # (N,) bool

    def at(self, times, max_gap_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated T_world_subject at `times`.

        Returns (M, 4, 4) poses and an (M,) validity mask. A time is invalid if it is
        outside the track, a neighbouring sample is occluded, or the samples around it
        are further apart than max_gap_s (dropped frames).
        """
        times = np.atleast_1d(np.asarray(times, float))
        if len(self.t) < 2:
            return np.full((len(times), 4, 4), np.nan), np.zeros(len(times), bool)
        i1 = np.clip(np.searchsorted(self.t, times), 1, len(self.t) - 1)
        i0 = i1 - 1
        t0, t1 = self.t[i0], self.t[i1]
        valid = (times >= self.t[0]) & (times <= self.t[-1]) & (t1 - t0 <= max_gap_s)
        valid &= ~self.occluded[i0] & ~self.occluded[i1]
        w = np.where(t1 > t0, (times - t0) / np.where(t1 > t0, t1 - t0, 1), 0.0)[:, None]

        pos = (1 - w) * self.pos[i0] + w * self.pos[i1]
        with np.errstate(invalid="ignore"):  # occluded samples are NaN and masked below
            q = slerp(self.quat[i0], self.quat[i1], w[:, 0])
        T = np.tile(np.eye(4), (len(times), 1, 1))
        ok = valid & np.isfinite(q).all(axis=1)
        T[ok, :3, :3] = quat_to_matrix(q[ok])
        T[ok, :3, 3] = pos[ok]
        T[~ok] = np.nan
        return T, ok


def slerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorised spherical interpolation of (N, 4) unit quaternions."""
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)  # shortest path
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1, 1))
    small = theta < 1e-6
    s = np.sin(theta)
    a = np.where(small, 1 - w, np.sin((1 - w) * theta) / np.where(small, 1, s))
    b = np.where(small, w, np.sin(w * theta) / np.where(small, 1, s))
    q = a[:, None] * q0 + b[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) x, y, z, w -> (N, 3, 3)."""
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def _read_csv(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))


class Recording:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path / "meta.yaml") as f:
            self.meta = yaml.safe_load(f)
        self.frames = _read_csv(self.path / "frames.csv")  # t_host, frame, stream_index, pts_ms
        self.video_path = self.path / "video.mkv"
        self.tracks: dict[str, PoseTrack] = {}
        if (self.path / "vicon_poses.csv").exists():
            self._load_poses(_read_csv(self.path / "vicon_poses.csv"))

    @property
    def calibration_path(self) -> Path | None:
        p = self.path / "calibration.yaml"
        return p if p.exists() else None

    @property
    def frame_times(self) -> np.ndarray:
        """Host receive time of every video frame. Subtract the camera delay to get exposure times."""
        return self.frames["t_host"].astype(float)

    def _load_poses(self, rows: np.ndarray) -> None:
        for subject in np.unique(rows["subject"]):
            r = np.sort(rows[rows["subject"] == subject], order="t_host")
            pos = np.column_stack([r["tx"], r["ty"], r["tz"]]).astype(float)
            quat = np.column_stack([r["qx"], r["qy"], r["qz"], r["qw"]]).astype(float)
            occ = r["occluded"].astype(bool) | ~np.isfinite(pos).all(axis=1)
            self.tracks[str(subject)] = PoseTrack(r["t_host"].astype(float), pos, quat, occ)

    def pose_at(self, subject: str, times, max_gap_s: float = 0.05):
        return self.tracks[subject].at(times, max_gap_s)

    def iter_frames(self, start: int = 0) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yields (frame index, t_host, BGR image)."""
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            i = start
            while i < len(self.frames):
                ok, img = cap.read()
                if not ok:
                    break
                yield i, float(self.frames["t_host"][i]), img
                i += 1
        finally:
            cap.release()
