"""Board pose from one ChArUco detection, valid for all lens models."""

from dataclasses import dataclass

import cv2
import numpy as np

from calib.board import Detection, object_points
from calib.intrinsics import project_points, undistort_points
from calib.transforms import from_rvec


@dataclass
class BoardPose:
    T_cam_board: np.ndarray  # (4, 4)
    rms_px: float
    observed: np.ndarray  # (N, 2) detected corners
    reprojected: np.ndarray  # (N, 2) corners projected with the fitted pose


def board_pose(board: cv2.aruco.CharucoBoard, det: Detection | None, model: str, K, dist,
               min_corners: int = 4) -> BoardPose | None:
    if det is None or len(det) < min_corners:
        return None
    obj = object_points(board, det)
    img = det.corners.reshape(-1, 2)
    # Solve in normalised coordinates so pinhole, rational and fisheye share one path.
    norm = undistort_points(model, img, K, dist)
    ok, rvec, tvec = cv2.solvePnP(obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, norm, np.eye(3), None, rvec, tvec)
    proj = project_points(model, obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
    return BoardPose(from_rvec(rvec, tvec), rms, img, proj)
