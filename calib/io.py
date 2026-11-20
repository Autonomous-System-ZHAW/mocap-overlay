"""Load and save calibration results as YAML.

Every result records when it was made and the camera settings it is valid for.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from calib.intrinsics import IntrinsicsResult


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_intrinsics(path: str | Path, res: IntrinsicsResult, camera: dict, board: dict) -> None:
    data = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_name": camera.get("name"),
        "camera_settings": camera.get("settings", {}),
        "board": board,
        "model": res.model,
        "image_size": list(res.image_size),
        "K": res.K.tolist(),
        "dist": res.dist.tolist(),
        "rms_px": round(res.rms_px, 4),
        "coverage": round(res.coverage, 3),
        "views": {n: round(float(e), 4) for n, e in zip(res.view_names, res.per_view_rms_px)},
        "rejected_views": res.rejected_views,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)


def load_intrinsics(path: str | Path) -> tuple[str, tuple[int, int], np.ndarray, np.ndarray]:
    """Returns (model, image_size, K, dist)."""
    d = load_yaml(path)
    return d["model"], tuple(d["image_size"]), np.array(d["K"], float), np.array(d["dist"], float)


# --- static pose sessions (extrinsics capture) -------------------------------------

def save_session_meta(session_dir: str | Path, meta: dict) -> None:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / "session.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=None)


def save_static_pose(session_dir: str | Path, name: str, det, T_world_body: np.ndarray, stats: dict) -> None:
    np.savez(Path(session_dir) / f"{name}.npz", corners=det.corners, ids=det.ids, T_world_body=T_world_body,
             **{k: np.asarray(v) for k, v in stats.items()})


def load_static_session(session_dir: str | Path):
    """Returns (meta dict, list[StaticPose]) sorted by name."""
    from calib.board import Detection
    from calib.extrinsics import StaticPose

    session_dir = Path(session_dir)
    meta = load_yaml(session_dir / "session.yaml")
    poses = []
    for f in sorted(session_dir.glob("pose_*.npz")):
        d = np.load(f)
        poses.append(StaticPose(f.stem, Detection(d["corners"], d["ids"]), d["T_world_body"]))
    return meta, poses


def save_extrinsics(path: str | Path, res, intrinsics_path: str | Path, camera: dict, board: dict,
                    session_dir: str | Path) -> None:
    """Self-contained result: includes the intrinsics it was solved with."""
    from calib.transforms import inv_T

    intr = load_yaml(intrinsics_path)
    data = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_name": camera.get("name"),
        "camera_settings": camera.get("settings", {}),
        "board": board,
        "session": str(session_dir),
        "intrinsics_file": str(intrinsics_path),
        "intrinsics": {k: intr[k] for k in ("model", "image_size", "K", "dist", "rms_px")},
        "T_cam_world": res.T_cam_world.tolist(),
        "T_world_cam": inv_T(res.T_cam_world).tolist(),
        "T_body_board": res.T_body_board.tolist(),
        "rms_px": round(res.rms_px, 4),
        "init_rms_px": round(res.init_rms_px, 4),
        "poses": {n: {"rms_px": round(e, 4), "t_err_mm": round(res.per_pose_err[n][0], 2),
                      "r_err_deg": round(res.per_pose_err[n][1], 3)} for n, e in res.per_pose_rms_px.items()},
        "rejected_poses": res.rejected,
        "warnings": res.warnings,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
