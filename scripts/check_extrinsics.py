"""Live validation of the camera calibration with Vicon data.

Draws into the camera image:
  - floor grid (z = 0) and the Vicon world axes at the origin
  - every Vicon marker (yellow circles): they must sit on the real markers
  - the ChArUco board as predicted by Vicon (cyan): must cover the printed board
  - the axes of all tracked objects

Place single markers or the wand all over the driving area and check the offset
everywhere in the image, especially near the image borders.

The camera delay (newest data/time_sync result, or --delay) is applied: each frame is
drawn with the Vicon data from the moment it was exposed, so the overlay stays on moving
objects. Press D to toggle the correction and see the difference.

Keys: G grid, D delay correction on/off, S snapshot, Q / ESC quit.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calib.board import BoardConfig
from calib.camera import Camera
from calib.io import load_yaml, load_vicon_config
from calib.time_sync import load_delay
from overlay import draw
from record.camera_stream import CameraStream, State as CameraState
from record.vicon_stream import ViconStream


def latest_file(folder: str, pattern: str) -> Path:
    files = sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no {pattern} in {folder}, run calibrate_extrinsics.py first or pass --extrinsics")
    return files[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--vicon", default="config/vicon.yaml")
    p.add_argument("--extrinsics", help="extrinsics YAML (default: newest in data/extrinsics)")
    p.add_argument("--source", help="override stream URL / device index / video file")
    p.add_argument("--grid", type=float, nargs=4, default=[-2.5, 2.5, -2.5, 2.5], metavar=("X0", "X1", "Y0", "Y1"),
                   help="floor grid extent in Vicon world [m]")
    p.add_argument("--grid-step", type=float, default=0.5)
    p.add_argument("--time-sync", help="time sync YAML (default: newest in data/time_sync)")
    p.add_argument("--delay", type=float, help="camera delay [s], overrides the time sync file")
    a = p.parse_args()

    ext_path = Path(a.extrinsics) if a.extrinsics else latest_file("data/extrinsics", "extrinsics_*.yaml")
    ext = load_yaml(ext_path)
    cam = Camera.from_extrinsics_yaml(ext_path)
    board = BoardConfig(**ext["board"]).build()
    T_body_board = np.array(ext["T_body_board"])
    print(f"using {ext_path} (RMS {ext['rms_px']} px)")
    delay, _, sync_path = load_delay(a.time_sync)
    if a.delay is not None:
        delay, sync_path = a.delay, "--delay"
    print(f"camera delay {delay * 1000:.0f} ms ({sync_path or 'not measured, run estimate_time_offset.py'})")

    cam_cfg, vicon_cfg = load_yaml(a.camera), load_vicon_config(a.vicon)
    board_subject = vicon_cfg["subjects"]["board"]
    subjects = list(dict.fromkeys(vicon_cfg["subjects"].values()))
    vicon = ViconStream(vicon_cfg["host"], vicon_cfg["sdk_dir"], subjects, vicon_cfg.get("stream_mode", "ServerPush"),
                        with_markers=True)
    show_grid, use_delay = True, True

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
            view = frame.copy()
            if show_grid:
                draw.floor_grid(view, cam, a.grid[:2], a.grid[2:], a.grid_step)
                draw.axes(view, cam, np.eye(4), length=0.5, thickness=4)

            t_exposure = grabbed.t_host - (delay if use_delay else 0.0)
            s = vicon.nearest(t_exposure)  # markers: nearest sample (not interpolable)
            if s is not None:
                for name in subjects:
                    T = vicon.pose_at(name, t_exposure)
                    if T is None:
                        continue
                    draw.axes(view, cam, T)
                    if name == board_subject:
                        draw.board_outline(view, cam, board, T @ T_body_board, (255, 255, 0))
                if s.markers is not None and len(s.markers):
                    draw.points(view, cam, s.markers, (0, 255, 255), radius=8)

            n_markers = 0 if s is None or s.markers is None else len(s.markers)
            text = (f"vicon {vicon.state.value}, {n_markers} markers   calib RMS {ext['rms_px']} px   "
                    f"delay {'%.0f ms' % (delay * 1000) if use_delay else 'OFF'}")
            cv2.putText(view, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            scale = min(1.0, 1600 / view.shape[1])
            cv2.imshow("check extrinsics", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_grid = not show_grid
            elif key == ord("d"):
                use_delay = not use_delay
            elif key == ord("s"):
                out = Path("data/extrinsics/check") / f"check_{datetime.now():%Y%m%d_%H%M%S}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), view)
                print(f"saved {out}")


if __name__ == "__main__":
    main()
