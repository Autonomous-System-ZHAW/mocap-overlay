"""Live check of an intrinsics calibration.

Left: original frame. Right: undistorted frame. With a good calibration, straight
lines in the room (door frames, shelves, floor tiles) are straight on the right,
also near the image corners. Toggle the grid to judge straightness.

If the ChArUco board is visible, its pose is fitted with the calibration and the
reprojection RMS is shown (should stay below about 0.5-1 px everywhere in the image).
Red lines show the residuals, magnified 20x.

Keys: G grid, [ / ] alpha (crop <-> full view), S snapshot, Q / ESC quit.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calib.board import BoardConfig, detect, make_detector
from calib.intrinsics import undistort_maps
from calib.pose import board_pose
from calib.io import load_intrinsics, load_yaml
from record.camera_stream import CameraStream, State as CameraState

RESIDUAL_GAIN = 20


def latest_calibration(folder: Path) -> Path:
    files = sorted(folder.glob("intrinsics_*.yaml"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no intrinsics_*.yaml in {folder}, run calibrate_intrinsics.py first or pass --calib")
    return files[-1]


def draw_grid(img: np.ndarray, step: int = 80) -> None:
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (255, 255, 0), 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--board", default="config/board.yaml")
    p.add_argument("--calib", help="intrinsics YAML (default: newest in data/intrinsics)")
    p.add_argument("--source", help="override stream URL / device index / video file")
    p.add_argument("--alpha", type=float, default=0.0)
    a = p.parse_args()

    calib_path = Path(a.calib) if a.calib else latest_calibration(Path("data/intrinsics"))
    model, size, K, dist = load_intrinsics(calib_path)
    print(f"using {calib_path} ({model}, {size[0]}x{size[1]})")

    cam = load_yaml(a.camera)
    board = BoardConfig.from_yaml(a.board).build()
    detector = make_detector(board)

    alpha, show_grid = a.alpha, True
    m1, m2, _ = undistort_maps(model, K, dist, size, alpha)
    snap_dir = Path("data/intrinsics/check")

    with CameraStream(cam, a.source) as camera:
        while True:
            grabbed = camera.read(timeout=1.0)
            if grabbed is None:
                if camera.state == CameraState.STOPPED:
                    print(f"camera stopped: {camera.last_error}")
                    break
                cv2.waitKey(1)  # keep the window responsive while reconnecting
                continue
            frame = grabbed.image
            if (frame.shape[1], frame.shape[0]) != size:
                raise SystemExit(f"frame is {frame.shape[1]}x{frame.shape[0]}, calibration is {size[0]}x{size[1]}")

            undist = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
            left = frame.copy()

            fit = board_pose(board, detect(detector, frame), model, K, dist)
            if fit is not None:
                rms, obs, proj = fit.rms_px, fit.observed, fit.reprojected
                for o, q in zip(obs, proj):
                    end = o + (q - o) * RESIDUAL_GAIN
                    cv2.circle(left, tuple(o.astype(int)), 4, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.line(left, tuple(o.astype(int)), tuple(end.astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
                color = (0, 255, 0) if rms < 1.0 else (0, 0, 255)
                cv2.putText(left, f"board RMS {rms:.2f} px", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

            if show_grid:
                draw_grid(undist)
            cv2.putText(left, "original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(undist, f"undistorted ({model}, alpha {alpha:.1f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            view = np.hstack([left, undist])
            scale = min(1.0, 1800 / view.shape[1])
            cv2.imshow("check intrinsics", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_grid = not show_grid
            elif key in (ord("["), ord("]")):
                alpha = float(np.clip(alpha + (0.1 if key == ord("]") else -0.1), 0.0, 1.0))
                m1, m2, _ = undistort_maps(model, K, dist, size, alpha)
            elif key == ord("s"):
                snap_dir.mkdir(parents=True, exist_ok=True)
                path = snap_dir / f"check_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), view)
                print(f"saved {path}")


if __name__ == "__main__":
    main()
