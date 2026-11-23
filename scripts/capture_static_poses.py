"""Capture static ChArUco board poses (camera + Vicon) for the extrinsics calibration.

Place the board, let go, press SPACE. After a short settle time the script records
camera detections and Vicon poses for --hold seconds and saves the pose only if the
board was still in both (corner jitter and Vicon motion below the limits).

Aim for ~20 poses spread over the whole driving area and the whole image, at
different distances and tilted about different axes (not only flat on the floor).

Keys: SPACE capture, U undo last pose, Q / ESC quit.
"""

import argparse
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from calib.board import BoardConfig, Detection, detect, effective_min_corners, make_detector
from calib.io import load_intrinsics, load_yaml, save_session_meta, save_static_pose, load_vicon_config
from calib.pose import board_pose
from calib.transforms import average_poses
from record.camera_stream import CameraStream, State as CameraState
from record.vicon_stream import State as ViconState
from record.vicon_stream import ViconStream


class Mode(Enum):
    LIVE = "live"
    SETTLING = "settling"
    COLLECTING = "collecting"


def latest_file(folder: str, pattern: str) -> Path:
    files = sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no {pattern} in {folder}")
    return files[-1]


def average_detections(dets):
    """Mean corners over frames, using only corner ids seen in every frame. Returns (det, max std px)."""
    common = set(dets[0].ids.ravel())
    for d in dets[1:]:
        common &= set(d.ids.ravel())
    ids = np.array(sorted(common), np.int32)
    if len(ids) == 0:
        return None, np.inf
    stack = []
    for d in dets:
        index = {int(i): k for k, i in enumerate(d.ids.ravel())}
        stack.append(d.corners.reshape(-1, 2)[[index[int(i)] for i in ids]])
    stack = np.stack(stack)
    std = float(np.linalg.norm(stack.std(axis=0), axis=1).max())
    return Detection(stack.mean(axis=0).reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1)), std


def next_pose_name(session: Path) -> str:
    i = 0
    while (session / f"pose_{i:03d}.npz").exists():
        i += 1
    return f"pose_{i:03d}"


def evaluate_capture(dets: list[Detection], samples, subject: str, a):
    """Check that the board was still in camera and Vicon during the recording window.

    Returns ((avg detection, T_world_body, stats), message) or (None, reason).
    """
    T_list = [s.poses.get(subject) for s in samples]
    if len(dets) < 3:
        return None, f"rejected: board seen in only {len(dets)} frames"
    if not samples or any(T is None for T in T_list):
        return None, "rejected: Vicon board missing/occluded during capture"
    avg_det, corner_std = average_detections(dets)
    T_wb, t_std, r_dev = average_poses(T_list)
    if avg_det is None or len(avg_det) < a.min_corners:
        return None, "rejected: too few corners visible in all frames"
    if corner_std > a.max_corner_std:
        return None, f"rejected: corners moved {corner_std:.2f} px"
    if t_std * 1000 > a.max_move_mm or r_dev > a.max_rot_deg:
        return None, f"rejected: Vicon moved {t_std * 1000:.2f} mm / {r_dev:.2f} deg"
    stats = {"corner_std_px": corner_std, "vicon_t_std_m": t_std, "vicon_r_dev_deg": r_dev,
             "n_frames": len(dets), "n_vicon": len(samples)}
    return (avg_det, T_wb, stats), (f"{len(avg_det)} corners, jitter {corner_std:.2f} px, "
                                    f"Vicon {t_std * 1000:.2f} mm / {r_dev:.2f} deg")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--board", default="config/board.yaml")
    p.add_argument("--vicon", default="config/vicon.yaml")
    p.add_argument("--intrinsics", help="intrinsics YAML (default: newest in data/intrinsics)")
    p.add_argument("--source", help="override stream URL / device index / video file")
    p.add_argument("--session", help="session folder to continue (default: new one in data/extrinsics)")
    p.add_argument("--min-corners", type=int, default=10, help="capped at the board's corner count")
    p.add_argument("--settle", type=float, default=0.8, help="wait after SPACE before recording [s]")
    p.add_argument("--hold", type=float, default=1.0, help="recording duration per pose [s]")
    p.add_argument("--max-corner-std", type=float, default=0.3, help="max corner jitter [px]")
    p.add_argument("--max-move-mm", type=float, default=0.5, help="max Vicon translation std [mm]")
    p.add_argument("--max-rot-deg", type=float, default=0.15, help="max Vicon rotation deviation [deg]")
    a = p.parse_args()

    cam_cfg, vicon_cfg = load_yaml(a.camera), load_vicon_config(a.vicon)
    board_cfg = BoardConfig.from_yaml(a.board)
    board = board_cfg.build()
    detector = make_detector(board)
    a.min_corners = effective_min_corners(board, a.min_corners)
    intr_path = Path(a.intrinsics) if a.intrinsics else latest_file("data/intrinsics", "intrinsics_*.yaml")
    model, size, K, dist = load_intrinsics(intr_path)
    subject = vicon_cfg["subjects"]["board"]

    session = Path(a.session) if a.session else Path("data/extrinsics") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if not (session / "session.yaml").exists():
        save_session_meta(session, {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "intrinsics_file": str(intr_path), "board": asdict(board_cfg), "vicon_subject": subject,
            "camera_settings": cam_cfg.get("settings", {}),
        })
    names = sorted(f.stem for f in session.glob("pose_*.npz"))
    print(f"session {session} ({len(names)} poses so far), intrinsics {intr_path}")
    history = []  # corners of saved poses, for the coverage display

    vicon = ViconStream(vicon_cfg["host"], vicon_cfg["sdk_dir"], [subject], vicon_cfg.get("stream_mode", "ServerPush"))
    mode, t_mode, dets, msg, msg_color = Mode.LIVE, 0.0, [], "", (255, 255, 255)

    with vicon, CameraStream(cam_cfg, a.source) as camera:
        while True:
            grabbed = camera.read(timeout=1.0)
            if grabbed is None:
                if camera.state == CameraState.STOPPED:
                    print(f"camera stopped: {camera.last_error}")
                    break
                cv2.waitKey(1)  # keep the window responsive while reconnecting
                continue
            frame = grabbed.image
            now = grabbed.t_host
            if (frame.shape[1], frame.shape[0]) != size:
                raise SystemExit(f"frame is {frame.shape[1]}x{frame.shape[0]}, intrinsics are {size[0]}x{size[1]}")
            det = detect(detector, frame)
            fit = board_pose(board, det, model, K, dist, min_corners=a.min_corners)

            # --- state machine ----------------------------------------------------
            if mode == Mode.SETTLING and now - t_mode >= a.settle:
                mode, t_mode, dets = Mode.COLLECTING, now, []
            if mode == Mode.COLLECTING:
                if det is not None:
                    dets.append(det)
                if now - t_mode >= a.hold:
                    mode = Mode.LIVE
                    result, msg = evaluate_capture(dets, vicon.window(t_mode - a.settle, now), subject, a)
                    ok_pose = result is not None
                    if ok_pose:
                        avg_det, T_wb, stats = result
                        name = next_pose_name(session)
                        save_static_pose(session, name, avg_det, T_wb, stats)
                        cv2.imwrite(str(session / f"{name}.png"), frame)
                        names.append(name)
                        history.append(avg_det)
                        msg = f"saved {name}: {msg}"
                    msg_color = (0, 255, 0) if ok_pose else (0, 0, 255)
                    print(msg)

            # --- display ------------------------------------------------------------
            view = frame.copy()
            for d in history:
                for x, y in d.corners.reshape(-1, 2):
                    cv2.circle(view, (int(x), int(y)), 4, (0, 200, 0), -1, cv2.LINE_AA)
            if det is not None:
                cv2.aruco.drawDetectedCornersCharuco(view, det.corners, det.ids, (0, 0, 255))
            latest = vicon.latest()
            v_ok = vicon.state == ViconState.STREAMING and latest is not None and latest.poses.get(subject) is not None
            lines = [
                (f"poses {len(names)}   mode {mode.value}", (255, 255, 255)),
                (f"camera: {0 if det is None else len(det)} corners" + (f", PnP {fit.rms_px:.2f} px" if fit else ""),
                 (0, 255, 0) if fit else (0, 0, 255)),
                (f"vicon: {vicon.state.value}, board {'tracked' if v_ok else 'NOT tracked'}",
                 (0, 255, 0) if v_ok else (0, 0, 255)),
                (msg, msg_color),
            ]
            for i, (text, color) in enumerate(lines):
                cv2.putText(view, text, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(view, text, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            scale = min(1.0, 1280 / view.shape[1])
            cv2.imshow("static poses", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and mode == Mode.LIVE:
                n_det = 0 if det is None else len(det)
                if fit is None:
                    msg, msg_color = f"not captured: camera sees {n_det}/{a.min_corners} corners", (0, 0, 255)
                elif not v_ok:
                    msg, msg_color = f"not captured: Vicon is {vicon.state.value}, board not tracked", (0, 0, 255)
                else:
                    mode, t_mode, msg, msg_color = Mode.SETTLING, now, "hold still ...", (0, 255, 255)
            if key == ord("u") and mode == Mode.LIVE and names:
                name = names.pop()
                for ext in (".npz", ".png"):
                    (session / f"{name}{ext}").unlink(missing_ok=True)
                if history:
                    history.pop()
                msg, msg_color = f"removed {name}", (0, 255, 255)
                print(msg)

    print(f"\n{len(names)} poses in {session}\nnext: uv run python scripts/calibrate_extrinsics.py {session}")


if __name__ == "__main__":
    main()
