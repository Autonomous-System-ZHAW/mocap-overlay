"""Synthetic check: the delay estimator recovers a known camera delay."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calib.board import BoardConfig
from calib.camera import Camera
from calib.time_sync import Corners, Problem, estimate
from calib.transforms import apply, make_T
from record.recording import PoseTrack

SIZE = (1920, 1080)
K = np.array([[1060.0, 0, 960], [0, 1060.0, 540], [0, 0, 1]])
DIST = np.array([-0.02, 0.03, -0.03, 0.01])  # fisheye


class FakeRecording:
    def __init__(self, track):
        self.track = track

    def pose_at(self, subject, times, max_gap_s=0.05):
        return self.track.at(times, max_gap_s)


def test_recovers_known_delay():
    board = BoardConfig(4, 3, 0.2445, 0.1835, "DICT_6X6_100").build()
    # Camera at the origin looking along world +y (OpenCV: x right, y down, z forward).
    T_cam_world = make_T(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), [0, 0, 0])
    cam = Camera("fisheye", SIZE, K, DIST, T_cam_world)
    T_body_board = make_T(np.eye(3), [-0.4, -0.3, 0.0])

    # Board 3 m in front of the camera, facing it, swinging around at up to ~1.5 m/s (200 Hz Vicon).
    t_v = np.arange(0, 30, 0.005)
    pos = np.column_stack([0.6 * np.sin(1.3 * t_v), 3 + 0.5 * np.sin(0.7 * t_v), 0.4 * np.sin(2.1 * t_v + 1)])
    R_face = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # board z axis points away from the camera
    rots = Rotation.from_matrix(R_face) * Rotation.from_rotvec(np.column_stack([0.2 * np.sin(t_v), 0.2 * np.cos(0.8 * t_v), 0 * t_v]))
    track = PoseTrack(t_v, pos, rots.as_quat(), np.zeros(len(t_v), bool))

    delay = 0.927
    t_frames = np.arange(1, 29, 0.04)  # 25 fps arrival times
    rng = np.random.default_rng(0)
    T_wb, ok = track.at(t_frames - delay)
    obj_all = board.getChessboardCorners().astype(float)
    frames, objs, imgs = [], [], []
    for i in np.flatnonzero(ok):
        px, vis = cam.project(apply(T_wb[i] @ T_body_board, obj_all))
        frames.append(np.full(vis.sum(), i))
        objs.append(obj_all[vis])
        imgs.append(px[vis] + rng.normal(scale=0.5, size=(vis.sum(), 2)))
    corners = Corners(np.concatenate(frames), np.concatenate(objs), np.concatenate(imgs))

    prob = Problem(FakeRecording(track), "board", cam, T_body_board, corners, t_frames)
    res = estimate(prob, "receive")
    assert abs(res.delay_s - delay) < 0.002
    assert res.rms_px < 1.0 < res.rms_px_no_delay
    assert all(abs(d - delay) < 0.005 for _, d in res.windows)
