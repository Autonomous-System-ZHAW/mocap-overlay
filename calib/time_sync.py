"""Camera delay from a recording of a moving ChArUco board that Vicon tracks.

Frames are stamped when they arrive on the host; the image was exposed `delay` earlier.
For a candidate delay d (and rolling-shutter readout time r), every detected corner is
predicted from the Vicon board pose at

    t_corner = t_frame - d + (y / H - 0.5) * r          (row y exposed later than row 0)

projected with the calibrated camera and compared with the detected corner. The (d, r)
with the smallest reprojection error is the answer.

Two frame time bases are compared:
  receive  host time when the frame arrived (includes network / decode jitter)
  pts      stream timestamp, mapped to the host clock with the earliest-arrival offset
           (removes the arrival jitter if the camera clock is stable)

A negative readout means the sensor is read bottom to top (e.g. image rotated 180 deg).
"""

from dataclasses import dataclass, field

import numpy as np

from calib.board import Detection, object_points
from calib.camera import Camera
from calib.intrinsics import project_points
from calib.transforms import to_rvec
from record.recording import Recording

CLIP_PX = 30.0  # errors above this count as outliers (wrong detection, occlusion)


@dataclass
class Corners:
    """All detected corners of a recording, one row per corner."""

    frame: np.ndarray  # (N,) frame index
    obj: np.ndarray  # (N, 3) board coordinates [m]
    img: np.ndarray  # (N, 2) detected pixel


@dataclass
class SyncResult:
    timebase: str
    delay_s: float
    readout_s: float
    rms_px: float  # at the optimum, inlier corners
    rms_px_no_delay: float  # same corners, delay 0 (what you get without correction)
    inlier_ratio: float
    curve: tuple[np.ndarray, np.ndarray]  # (delays, cost) at the best readout
    windows: list[tuple[float, float]] = field(default_factory=list)  # (window start [s], delay [s])


def collect_corners(dets: dict[int, Detection], board) -> Corners:
    frames, objs, imgs = [], [], []
    for i, det in sorted(dets.items()):
        frames.append(np.full(len(det), i))
        objs.append(object_points(board, det))
        imgs.append(det.corners.reshape(-1, 2))
    return Corners(np.concatenate(frames), np.concatenate(objs).astype(float), np.concatenate(imgs).astype(float))


def frame_times(rec: Recording, timebase: str) -> np.ndarray:
    t_recv = rec.frame_times
    if timebase == "receive":
        return t_recv
    pts = rec.frames["pts_ms"].astype(float) / 1000
    if not np.isfinite(pts).all():
        raise ValueError("recording has frames without pts")
    # The earliest arrivals had the least network/decode delay: map pts onto the host clock with them.
    return pts + np.percentile(t_recv - pts, 2)


class Problem:
    def __init__(self, rec: Recording, subject: str, cam: Camera, T_body_board: np.ndarray, corners: Corners,
                 t_frame: np.ndarray):
        self.rec, self.subject, self.cam, self.T_body_board = rec, subject, cam, T_body_board
        self.c = corners
        self.t_frame = t_frame[corners.frame]
        self.row = corners.img[:, 1] / cam.image_size[1] - 0.5

    def errors(self, delay: float, readout: float = 0.0) -> np.ndarray:
        """Per-corner reprojection error [px]; NaN where Vicon has no valid pose."""
        t = self.t_frame - delay + self.row * readout
        T_wb, ok = self.rec.pose_at(self.subject, t)
        err = np.full(len(t), np.nan)
        if not ok.any():
            return err
        T_cb = self.cam.T_cam_world @ T_wb[ok] @ self.T_body_board  # (M, 4, 4)
        pts_cam = np.einsum("nij,nj->ni", T_cb[:, :3, :3], self.c.obj[ok]) + T_cb[:, :3, 3]
        rvec, tvec = to_rvec(np.eye(4))
        px = project_points(self.cam.model, pts_cam, rvec, tvec, self.cam.K, self.cam.dist)
        err[ok] = np.linalg.norm(px - self.c.img[ok], axis=1)
        return err

    def cost(self, delay: float, readout: float = 0.0, mask: np.ndarray | None = None) -> float:
        e = self.errors(delay, readout)
        if mask is not None:
            e = e[mask]
        return float(np.mean(np.minimum(np.nan_to_num(e, nan=CLIP_PX), CLIP_PX)))


def _minimise_1d(f, lo: float, hi: float, step: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Grid search, then 1 ms grid around the minimum, then a parabola through the best 3 points."""
    xs = np.arange(lo, hi + step / 2, step)
    ys = np.array([f(x) for x in xs])
    x0 = xs[np.argmin(ys)]
    fine = np.arange(x0 - step, x0 + step + 1e-4, 0.001)
    fy = np.array([f(x) for x in fine])
    k = int(np.clip(np.argmin(fy), 1, len(fine) - 2))
    y0, y1, y2 = fy[k - 1: k + 2]
    denom = y0 - 2 * y1 + y2
    x_best = fine[k] + (0.5 * (y0 - y2) / denom * 0.001 if denom > 0 else 0.0)
    return float(x_best), xs, ys


def estimate(problem: Problem, timebase: str, delay_range=(-0.5, 2.0), readout_range=(-0.04, 0.04),
             window_s: float = 5.0) -> SyncResult:
    # 1) delay without rolling shutter
    d, xs, ys = _minimise_1d(lambda x: problem.cost(x), *delay_range, step=0.01)
    # 2) readout time at that delay, then the delay again with the readout
    readouts = np.arange(readout_range[0], readout_range[1] + 1e-9, 0.002)
    costs = [problem.cost(d, x) for x in readouts]
    r = float(readouts[np.argmin(costs)])
    if min(costs) > 0.98 * problem.cost(d, 0.0):
        r = 0.0  # rolling shutter not measurable in this recording (needs fast vertical image motion)
    d, _, _ = _minimise_1d(lambda x: problem.cost(x, r), d - 0.03, d + 0.03, step=0.005)

    e = problem.errors(d, r)
    inl = np.isfinite(e) & (e < CLIP_PX)
    e0 = problem.errors(0.0, 0.0)
    both = inl & np.isfinite(e0)

    # 3) stability: delay per time window (readout fixed)
    windows = []
    t_rel = problem.t_frame - problem.t_frame.min()
    for w0 in np.arange(0, t_rel.max() - window_s / 2, window_s):
        m = (t_rel >= w0) & (t_rel < w0 + window_s)
        if m.sum() < 30:
            continue
        dw, _, _ = _minimise_1d(lambda x: problem.cost(x, r, m), d - 0.1, d + 0.1, step=0.01)
        windows.append((float(w0), dw))

    return SyncResult(
        timebase=timebase, delay_s=d, readout_s=r,
        rms_px=float(np.sqrt(np.mean(e[inl] ** 2))),
        rms_px_no_delay=float(np.sqrt(np.mean(np.minimum(e0[both], 1e3) ** 2))),
        inlier_ratio=float(inl.mean()), curve=(xs, ys), windows=windows,
    )


def load_delay(path=None) -> tuple[float, float, str | None]:
    """(delay_s, readout_s, file) from a time sync result; newest in data/time_sync by default.
    Returns zeros and None if nothing has been measured yet."""
    from pathlib import Path

    from calib.io import load_yaml

    if path is None:
        files = sorted(Path("data/time_sync").glob("time_sync_*.yaml"))
        if not files:
            return 0.0, 0.0, None
        path = files[-1]
    d = load_yaml(path)
    return float(d["delay_s"]), float(d.get("readout_s", 0.0)), str(path)
