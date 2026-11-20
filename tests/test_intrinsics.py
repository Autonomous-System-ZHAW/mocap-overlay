"""Synthetic round trip: render the board through a known camera, calibrate, compare."""

import cv2
import numpy as np
import pytest

from calib.board import BoardConfig, detect, make_detector, render_board
from calib.intrinsics import calibrate

SIZE = (1280, 720)
K_TRUE = np.array([[900.0, 0, 645.0], [0, 900.0, 355.0], [0, 0, 1]])
DIST_TRUE = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
PX_PER_M = 4000.0


@pytest.fixture(scope="module")
def board_setup():
    cfg = BoardConfig(squares_x=7, squares_y=5, square_length_m=0.06, marker_length_m=0.045)
    board = cfg.build()
    return board, render_board(board, PX_PER_M)


def synth_view(board_img, rvec, tvec, K, dist, size):
    """Render the planar board as seen by a distorted pinhole camera."""
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-6)
    norm = cv2.undistortPoints(pix, K, dist, criteria=crit).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    H = np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])  # board plane (X, Y, 1) -> normalised image
    XY = np.linalg.solve(H, np.column_stack([norm, np.ones(len(norm))]).T)
    X, Y = XY[0] / XY[2], XY[1] / XY[2]
    mx = (X * PX_PER_M - 0.5).reshape(h, w).astype(np.float32)
    my = (Y * PX_PER_M - 0.5).reshape(h, w).astype(np.float32)
    img = cv2.remap(board_img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


def random_poses(n, rng, board_center):
    poses = []
    for _ in range(n):
        rvec = rng.uniform([-0.6, -0.6, -0.4], [0.6, 0.6, 0.4])
        R, _ = cv2.Rodrigues(rvec)
        dist_m = rng.uniform(0.6, 1.0)
        # Place the board centre somewhere across the image so corners get covered.
        target = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.25, 0.25), 1.0]) * dist_m
        tvec = target - R @ board_center
        poses.append((rvec, tvec))
    return poses


def test_pinhole_recovers_known_camera(board_setup):
    board, board_img = board_setup
    detector = make_detector(board)
    sx, sy = board.getChessboardSize()
    center = np.array([sx, sy, 0]) * board.getSquareLength() / 2
    rng = np.random.default_rng(0)

    detections = {}
    for i, (r, t) in enumerate(random_poses(25, rng, center)):
        img = synth_view(board_img, r, t, K_TRUE, DIST_TRUE, SIZE)
        det = detect(detector, img)
        if det is not None:
            detections[f"v{i}"] = det
    assert len(detections) >= 15

    res = calibrate(board, detections, SIZE, model="pinhole")
    assert res.rms_px < 0.5
    np.testing.assert_allclose(np.diag(res.K)[:2], np.diag(K_TRUE)[:2], rtol=0.01)
    np.testing.assert_allclose(res.K[:2, 2], K_TRUE[:2, 2], atol=5.0)
    assert abs(res.dist[0] - DIST_TRUE[0]) < 0.03


def test_detection_shapes_are_normalised(board_setup):
    board, board_img = board_setup
    det = detect(make_detector(board), board_img)
    n = len(det)
    assert det.corners.shape == (n, 1, 2) and det.ids.shape == (n, 1)
    view = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedCornersCharuco(view, det.corners, det.ids, (0, 0, 255))  # raised on OpenCV 5


def test_undistorted_view_is_pinhole(board_setup):
    """After remapping with the true calibration the board fits a distortion-free camera."""
    from calib.intrinsics import undistort_maps

    board, board_img = board_setup
    sx, sy = board.getChessboardSize()
    rvec = np.array([0.3, -0.4, 0.1])
    tvec = np.array([0.35, 0.15, 0.75]) - cv2.Rodrigues(rvec)[0] @ (np.array([sx, sy, 0]) * 0.03)  # board near a corner
    img = synth_view(board_img, rvec, tvec, K_TRUE, DIST_TRUE, SIZE)
    m1, m2, K_new = undistort_maps("pinhole", K_TRUE, DIST_TRUE, SIZE, alpha=0.0)
    det = detect(make_detector(board), cv2.remap(img, m1, m2, cv2.INTER_LINEAR))
    assert det is not None and len(det) >= 15

    obj = board.getChessboardCorners()[det.ids.ravel()]
    _, r, t = cv2.solvePnP(obj, det.corners, K_new, None)
    proj, _ = cv2.projectPoints(obj, r, t, K_new, None)
    rms = np.sqrt(np.mean(np.sum((proj - det.corners).reshape(-1, 2) ** 2, axis=1)))
    assert rms < 0.5


def test_board_pose_rms_with_true_calibration(board_setup):
    from calib.pose import board_pose

    board, board_img = board_setup
    sx, sy = board.getChessboardSize()
    rvec = np.array([-0.2, 0.3, 0.0])
    tvec = np.array([-0.3, -0.1, 0.8]) - cv2.Rodrigues(rvec)[0] @ (np.array([sx, sy, 0]) * 0.03)
    img = synth_view(board_img, rvec, tvec, K_TRUE, DIST_TRUE, SIZE)
    det = detect(make_detector(board), img)
    fit_true = board_pose(board, det, "pinhole", K_TRUE, DIST_TRUE)
    fit_none = board_pose(board, det, "pinhole", K_TRUE, np.zeros(5))
    assert fit_true.rms_px < 0.5 < fit_none.rms_px  # ignoring the distortion must show up as error
    np.testing.assert_allclose(fit_true.T_cam_board[:3, 3], tvec, atol=0.005)
