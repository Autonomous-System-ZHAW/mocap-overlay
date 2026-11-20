"""Synthetic check of the Vicon -> camera extrinsics solver."""

import cv2
import numpy as np

from calib.board import BoardConfig, Detection
from calib.camera import Camera
from calib.extrinsics import StaticPose, calibrate, rotation_diversity
from calib.intrinsics import project_points
from calib.transforms import apply, from_rvec, inv_T, make_T, rotation_angle_deg, to_rvec

SIZE = (1920, 1080)
K = np.array([[1100.0, 0, 960], [0, 1100.0, 540], [0, 0, 1]])
DIST = np.array([-0.3, 0.1, 0.0, 0.0, -0.02])


def look_at(cam_pos, target, up=(0, 0, 1)) -> np.ndarray:
    """T_cam_world for an OpenCV camera (x right, y down, z forward) at cam_pos looking at target."""
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_world_cam = np.column_stack([x, y, z])
    return inv_T(make_T(R_world_cam, cam_pos))


def make_poses(board, T_cw, T_bb, n, rng, vicon_noise=(0.0002, 0.05), px_noise=0.2):
    sx, sy = board.getChessboardSize()
    center = np.array([sx, sy, 0]) * board.getSquareLength() / 2
    all_obj = board.getChessboardCorners()
    poses = []
    while len(poses) < n:
        # Board facing the camera, tilted up to ~50 degrees about random axes.
        tilt = rng.normal(size=3)
        tilt *= np.radians(rng.uniform(0, 50)) / np.linalg.norm(tilt)
        R_cb = cv2.Rodrigues(tilt)[0]
        pos = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3), 1.0]) * rng.uniform(2.0, 6.0)
        T_cb = make_T(R_cb, pos - R_cb @ center)
        rvec, tvec = to_rvec(T_cb)
        px = project_points("pinhole", all_obj, rvec, tvec, K, DIST)
        inside = (px[:, 0] > 5) & (px[:, 0] < SIZE[0] - 5) & (px[:, 1] > 5) & (px[:, 1] < SIZE[1] - 5)
        if inside.sum() < 12:
            continue
        ids = np.flatnonzero(inside)
        corners = px[ids] + rng.normal(scale=px_noise, size=(len(ids), 2))
        T_wb = inv_T(T_cw) @ T_cb @ inv_T(T_bb)
        noise = from_rvec(rng.normal(scale=np.radians(vicon_noise[1]), size=3), rng.normal(scale=vicon_noise[0], size=3))
        det = Detection(corners.reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1).astype(np.int32))
        poses.append(StaticPose(f"p{len(poses):02d}", det, T_wb @ noise))
    return poses


def test_recovers_camera_and_board_offset():
    board = BoardConfig(squares_x=7, squares_y=5, square_length_m=0.0735, marker_length_m=0.055,
                        dictionary="DICT_6X6_100").build()
    rng = np.random.default_rng(3)
    T_cw = look_at(np.array([-3.0, -3.0, 2.5]), np.array([0.5, 0.5, 0.0]))
    T_bb = make_T(cv2.Rodrigues(np.array([0.1, -0.2, 1.5]))[0], [0.12, -0.05, 0.02])

    poses = make_poses(board, T_cw, T_bb, 20, rng)
    # One pose with a wrong Vicon pose (e.g. a marker swapped): must be rejected.
    poses[5].T_world_body = poses[5].T_world_body @ make_T(np.eye(3), [0.03, 0, 0])

    res = calibrate(board, poses, "pinhole", K, DIST)

    assert res.rejected == ["p05"]
    assert res.rms_px < 0.5
    assert rotation_angle_deg(res.T_cam_world[:3, :3].T @ T_cw[:3, :3]) < 0.05
    cam_pos_err = np.linalg.norm(inv_T(res.T_cam_world)[:3, 3] - inv_T(T_cw)[:3, 3])
    assert cam_pos_err < 0.005
    np.testing.assert_allclose(res.T_body_board[:3, 3], T_bb[:3, 3], atol=0.002)

    # The resulting camera projects world points like the true one.
    cam = Camera("pinhole", SIZE, K, DIST, res.T_cam_world)
    pts = np.array([[x, y, 0.0] for x in np.linspace(-2, 2, 5) for y in np.linspace(-2, 2, 5)])
    px, ok = cam.project(pts)
    px_true, ok_true = Camera("pinhole", SIZE, K, DIST, T_cw).project(pts)
    assert (ok == ok_true).all() and ok.sum() > 10
    assert np.nanmax(np.linalg.norm(px[ok] - px_true[ok], axis=1)) < 1.0


def test_camera_masks_points_behind_and_outside():
    T_cw = look_at(np.array([0.0, -3.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    cam = Camera("pinhole", SIZE, K, DIST, T_cw)
    _, ok = cam.project(np.array([[0, 0, 1.0], [0, -6, 1.0], [30, -2.9, 1.0]]))  # front, behind, far off-axis
    assert ok.tolist() == [True, False, False]


def test_rotation_diversity_detects_single_axis():
    rng = np.random.default_rng(0)
    yaw_only = [make_T(cv2.Rodrigues(np.array([0, 0, a]))[0], [0, 0, 0]) for a in rng.uniform(-3, 3, 10)]
    varied = [make_T(cv2.Rodrigues(rng.normal(size=3))[0], [0, 0, 0]) for _ in range(10)]
    assert rotation_diversity(yaw_only) < 0.05
    assert rotation_diversity(varied) > 0.3


def test_apply_and_inverse():
    T = from_rvec([0.3, -0.2, 0.5], [1, 2, 3])
    p = np.array([[0.5, -1.0, 2.0]])
    np.testing.assert_allclose(apply(inv_T(T), apply(T, p)), p, atol=1e-12)


def test_session_round_trip(tmp_path):
    from calib.io import load_static_session, save_session_meta, save_static_pose

    det = Detection(np.random.rand(5, 1, 2).astype(np.float32), np.arange(5, dtype=np.int32).reshape(-1, 1))
    T = from_rvec([0.1, 0.2, 0.3], [1, 2, 3])
    save_session_meta(tmp_path, {"board": {"squares_x": 7}})
    save_static_pose(tmp_path, "pose_000", det, T, {"corner_std_px": 0.1})
    meta, poses = load_static_session(tmp_path)
    assert meta["board"]["squares_x"] == 7 and poses[0].name == "pose_000"
    np.testing.assert_allclose(poses[0].T_world_body, T)
    np.testing.assert_array_equal(poses[0].det.ids, det.ids)
