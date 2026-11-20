"""Camera intrinsics from ChArUco detections (pinhole, rational or fisheye model)."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from calib.board import Detection, object_points

MODELS = ("pinhole", "rational", "fisheye")


@dataclass
class IntrinsicsResult:
    model: str
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (N,) distortion coefficients, layout depends on model
    rms_px: float
    per_view_rms_px: np.ndarray  # (V,)
    view_names: list[str]  # views used in the final solve
    rejected_views: list[str] = field(default_factory=list)
    coverage: float = 0.0  # fraction of image grid cells containing at least one corner


def project_points(model: str, obj: np.ndarray, rvec, tvec, K, dist) -> np.ndarray:
    """Project 3D points (N, 3) given in the object frame to pixels (N, 2)."""
    if model == "fisheye":
        pts, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, dist)
    else:
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return pts.reshape(-1, 2)


def _solve(model: str, objs, imgs, image_size):
    if model == "fisheye":
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_FIX_SKEW
            | cv2.fisheye.CALIB_CHECK_COND
        )
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
        _, K, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3).astype(np.float64) for o in objs],
            [i.reshape(-1, 1, 2).astype(np.float64) for i in imgs],
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
    else:
        flags = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
        _, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, image_size, None, None, flags=flags)
    return K, dist.ravel(), rvecs, tvecs


def _per_view_rms(model, objs, imgs, rvecs, tvecs, K, dist) -> np.ndarray:
    errs = []
    for o, i, r, t in zip(objs, imgs, rvecs, tvecs):
        d = project_points(model, o, r, t, K, dist) - i.reshape(-1, 2)
        errs.append(np.sqrt(np.mean(np.sum(d**2, axis=1))))
    return np.array(errs)


def _total_rms(model, objs, imgs, rvecs, tvecs, K, dist) -> float:
    sq = [
        np.sum((project_points(model, o, r, t, K, dist) - i.reshape(-1, 2)) ** 2, axis=1)
        for o, i, r, t in zip(objs, imgs, rvecs, tvecs)
    ]
    return float(np.sqrt(np.mean(np.concatenate(sq))))


def coverage(detections: list[Detection], image_size: tuple[int, int], grid: int = 8) -> float:
    """Fraction of a grid x grid image tiling that contains at least one corner."""
    w, h = image_size
    hit = np.zeros((grid, grid), bool)
    for det in detections:
        xy = det.corners.reshape(-1, 2)
        gx = np.clip((xy[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((xy[:, 1] / h * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def calibrate(
    board: cv2.aruco.CharucoBoard,
    detections: dict[str, Detection],
    image_size: tuple[int, int],
    model: str = "pinhole",
    min_corners: int = 8,
    outlier_factor: float = 3.0,
) -> IntrinsicsResult:
    """Calibrate from named detections. Drops outlier views once and re-solves."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    views = {n: d for n, d in detections.items() if len(d) >= min_corners}
    if len(views) < 5:
        raise ValueError(f"need at least 5 views with >= {min_corners} corners, have {len(views)}")

    rejected: list[str] = []
    for attempt in range(2):
        names = list(views)
        objs = [object_points(board, views[n]) for n in names]
        imgs = [views[n].corners.reshape(-1, 2) for n in names]
        K, dist, rvecs, tvecs = _solve(model, objs, imgs, image_size)
        per_view = _per_view_rms(model, objs, imgs, rvecs, tvecs, K, dist)

        if attempt == 0 and outlier_factor > 0:
            limit = outlier_factor * np.median(per_view)
            bad = [n for n, e in zip(names, per_view) if e > limit]
            if bad and len(views) - len(bad) >= 5:
                rejected = bad
                views = {n: d for n, d in views.items() if n not in bad}
                continue
        break

    return IntrinsicsResult(
        model=model,
        image_size=tuple(image_size),
        K=K,
        dist=dist,
        rms_px=_total_rms(model, objs, imgs, rvecs, tvecs, K, dist),
        per_view_rms_px=per_view,
        view_names=names,
        rejected_views=rejected,
        coverage=coverage(list(views.values()), image_size),
    )


def undistort_maps(model: str, K, dist, image_size: tuple[int, int], alpha: float = 0.0):
    """Remap tables for cv2.remap and the new camera matrix of the undistorted image.

    alpha 0 crops to valid pixels only, 1 keeps the whole original image (black borders).
    For fisheye it is used as the `balance` parameter.
    """
    if model == "fisheye":
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist, image_size, np.eye(3), balance=alpha)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), K_new, image_size, cv2.CV_16SC2)
    else:
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, alpha, image_size)
        m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, image_size, cv2.CV_16SC2)
    return m1, m2, K_new


def undistort_points(model: str, pts: np.ndarray, K, dist) -> np.ndarray:
    """Pixels (N, 2) -> normalised image coordinates (N, 2), iterated to convergence."""
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    if model == "fisheye":
        out = cv2.fisheye.undistortPoints(pts, K, dist)
    else:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-8)
        out = cv2.undistortPoints(pts, K, dist, criteria=crit)
    return out.reshape(-1, 2)
