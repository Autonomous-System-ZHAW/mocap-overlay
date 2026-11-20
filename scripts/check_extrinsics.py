"""Live validation of the camera calibration with Vicon data.

Draws into the camera image:
  - floor grid (z = 0) and the Vicon world axes at the origin
  - every Vicon marker (yellow circles): they must sit on the real markers
  - the ChArUco board as predicted by Vicon (cyan): must cover the printed board
  - the axes of all tracked objects

Place single markers or the wand all over the driving area and check the offset
everywhere in the image, especially near the image borders.

Keys: G grid, S snapshot, Q / ESC quit.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calib.board import BoardConfig
from calib.camera import Camera
from calib.io import load_yaml
from overlay import draw
from record.stream import capture
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
    a = p.parse_args()

    ext_path = Path(a.extrinsics) if a.extrinsics else latest_file("data/extrinsics", "extrinsics_*.yaml")
    ext = load_yaml(ext_path)
    cam = Camera.from_extrinsics_yaml(ext_path)
    board = BoardConfig(**ext["board"]).build()
    T_body_board = np.array(ext["T_body_board"])
    print(f"using {ext_path} (RMS {ext['rms_px']} px)")

    cam_cfg, vicon_cfg = load_yaml(a.camera), load_yaml(a.vicon)
    board_subject = vicon_cfg["subjects"]["board"]
    subjects = list(dict.fromkeys(vicon_cfg["subjects"].values()))
    vicon = ViconStream(vicon_cfg["host"], vicon_cfg["sdk_dir"], subjects, vicon_cfg.get("stream_mode", "ServerPush"),
                        with_markers=True)
    show_grid = True

    with vicon, capture(cam_cfg, a.source) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("stream ended or dropped")
                break
            view = frame.copy()
            if show_grid:
                draw.floor_grid(view, cam, a.grid[:2], a.grid[2:], a.grid_step)
                draw.axes(view, cam, np.eye(4), length=0.5, thickness=4)

            s = vicon.latest()
            if s is not None:
                for name, T in s.poses.items():
                    if T is None:
                        continue
                    draw.axes(view, cam, T)
                    if name == board_subject:
                        draw.board_outline(view, cam, board, T @ T_body_board, (255, 255, 0))
                if s.markers is not None and len(s.markers):
                    draw.points(view, cam, s.markers, (0, 255, 255), radius=8)

            n_markers = 0 if s is None or s.markers is None else len(s.markers)
            text = f"vicon {vicon.state.value}, {n_markers} markers   calib RMS {ext['rms_px']} px"
            cv2.putText(view, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            scale = min(1.0, 1600 / view.shape[1])
            cv2.imshow("check extrinsics", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_grid = not show_grid
            elif key == ord("s"):
                out = Path("data/extrinsics/check") / f"check_{datetime.now():%Y%m%d_%H%M%S}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), view)
                print(f"saved {out}")


if __name__ == "__main__":
    main()
