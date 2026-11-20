"""Vicon world -> camera extrinsics from static ChArUco board poses (robot-world hand-eye).

Per static pose i we know
    T_cam_board_i   from the image (PnP on the ChArUco corners)
    T_world_body_i  from Vicon (the board's rigid body)
and solve for the two constant unknowns
    T_cam_world     camera pose in the Vicon world
    T_body_board    Vicon body frame -> printed board frame (no need to measure it)
from  T_cam_world · T_world_body_i · T_body_board = T_cam_board_i.

1. Initial guess: cv2.calibrateRobotWorldHandEye on the per-pose PnP results.
2. Refinement: least squares on the reprojection error of all corners of all poses.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from calib.board import Detection, object_points
from calib.intrinsics import project_points
from calib.pose import board_pose
from calib.transforms import from_rvec, inv_T, rotation_angle_deg, to_rvec


@dataclass
class StaticPose:
    name: str
    det: Detection
    T_world_body: np.ndarray  # (4, 4) averaged Vicon pose of the board's rigid body


@dataclass
class ExtrinsicsResult:
    T_cam_world: np.ndarray
    T_body_board: np.ndarray
    rms_px: float
    init_rms_px: float
    per_pose_rms_px: dict[str, float]
    per_pose_err: dict[str, tuple[float, float]]  # name -> (translation mm, rotation deg) vs. PnP pose
    rejected: list[str] = field(default_factory=list)
    residuals: np.ndarray | None = None  # (N, 4): observed x, y and residual dx, dy [px]
    warnings: list[str] = field(default_factory=list)


def initial_guess(T_cam_boards: list[np.ndarray], T_world_bodies: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form T_cam_world, T_body_board.

    OpenCV solves  T_cam_pattern_i · T_pattern_base = T_cam_gripper · T_gripper_base_i.
    Ours rearranged: T_cam_board_i · T_board_body = T_cam_world · T_world_body_i, so
    pattern = board, base = body, gripper = Vicon world.
    """
    R_board_body, t_board_body, R_cam_world, t_cam_world = cv2.calibrateRobotWorldHandEye(
        [T[:3, :3] for T in T_cam_boards], [T[:3, 3].reshape(3, 1) for T in T_cam_boards],
        [T[:3, :3] for T in T_world_bodies], [T[:3, 3].reshape(3, 1) for T in T_world_bodies],
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    T_board_body = np.eye(4)
    T_board_body[:3, :3], T_board_body[:3, 3] = R_board_body, t_board_body.ravel()
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3], T_cam_world[:3, 3] = R_cam_world, t_cam_world.ravel()
    return T_cam_world, inv_T(T_board_body)


def rotation_diversity(T_world_bodies: list[np.ndarray]) -> float:
    """How well the relative board rotations span 3D: ratio of the 2nd to 1st singular
    value of the rotation axes (0 = all about one axis -> degenerate, ~1 = good)."""
    axes = []
    for i in range(len(T_world_bodies)):
        for j in range(i + 1, len(T_world_bodies)):
            R = T_world_bodies[i][:3, :3].T @ T_world_bodies[j][:3, :3]
            if rotation_angle_deg(R) > 10:
                r = cv2.Rodrigues(R)[0].ravel()
                axes.append(r / np.linalg.norm(r))
    if len(axes) < 2:
        return 0.0
    s = np.linalg.svd(np.abs(np.array(axes)), compute_uv=False)  # axis sign is irrelevant
    return float(s[1] / s[0])


def _residuals(params, T0_cw, T0_bb, poses, objs, model, K, dist):
    T_cw = from_rvec(params[0:3], params[3:6]) @ T0_cw
    T_bb = from_rvec(params[6:9], params[9:12]) @ T0_bb
    out = []
    for p, obj in zip(poses, objs):
        rvec, tvec = to_rvec(T_cw @ p.T_world_body @ T_bb)
        out.append((project_points(model, obj, rvec, tvec, K, dist) - p.det.corners.reshape(-1, 2)).ravel())
    return np.concatenate(out)


def _per_pose_rms(res: np.ndarray, poses, objs) -> dict[str, float]:
    out, i = {}, 0
    for p, obj in zip(poses, objs):
        r = res[i: i + 2 * len(obj)].reshape(-1, 2)
        out[p.name] = float(np.sqrt(np.mean(np.sum(r**2, axis=1))))
        i += 2 * len(obj)
    return out


def calibrate(board: cv2.aruco.CharucoBoard, poses: list[StaticPose], model: str, K, dist,
              robust_scale_px: float = 1.0, outlier_px: float = 2.0) -> ExtrinsicsResult:
    warnings = []
    fits = {p.name: board_pose(board, p.det, model, K, dist) for p in poses}
    poses = [p for p in poses if fits[p.name] is not None]
    if len(poses) < 3:
        raise ValueError(f"need at least 3 usable poses, have {len(poses)}")
    div = rotation_diversity([p.T_world_body for p in poses])
    if div < 0.2:
        warnings.append(f"board rotations are nearly all about one axis (diversity {div:.2f}); "
                        "tilt the board about different axes, the solution may be poorly constrained")

    T0_cw, T0_bb = initial_guess([fits[p.name].T_cam_board for p in poses], [p.T_world_body for p in poses])

    rejected: list[str] = []
    for _ in range(2):
        objs = [object_points(board, p.det) for p in poses]
        args = (T0_cw, T0_bb, poses, objs, model, K, dist)
        x0 = np.zeros(12)
        init_res = _residuals(x0, *args)
        sol = least_squares(_residuals, x0, args=args, loss="soft_l1", f_scale=robust_scale_px, x_scale="jac")
        per_pose = _per_pose_rms(sol.fun, poses, objs)
        limit = max(outlier_px, 3 * float(np.median(list(per_pose.values()))))
        bad = [n for n, e in per_pose.items() if e > limit]
        if not bad or rejected or len(poses) - len(bad) < 3:
            break
        rejected = bad
        poses = [p for p in poses if p.name not in bad]

    T_cw = from_rvec(sol.x[0:3], sol.x[3:6]) @ T0_cw
    T_bb = from_rvec(sol.x[6:9], sol.x[9:12]) @ T0_bb

    per_pose_err = {}
    for p in poses:
        D = inv_T(fits[p.name].T_cam_board) @ T_cw @ p.T_world_body @ T_bb
        per_pose_err[p.name] = (float(np.linalg.norm(D[:3, 3]) * 1000), rotation_angle_deg(D[:3, :3]))

    obs = np.concatenate([p.det.corners.reshape(-1, 2) for p in poses])
    res = sol.fun.reshape(-1, 2)
    return ExtrinsicsResult(
        T_cam_world=T_cw,
        T_body_board=T_bb,
        rms_px=float(np.sqrt(np.mean(np.sum(res**2, axis=1)))),
        init_rms_px=float(np.sqrt(np.mean(np.sum(init_res.reshape(-1, 2) ** 2, axis=1)))),
        per_pose_rms_px=per_pose,
        per_pose_err=per_pose_err,
        rejected=rejected,
        residuals=np.hstack([obs, res]),
        warnings=warnings,
    )
