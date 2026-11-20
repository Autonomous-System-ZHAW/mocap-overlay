"""Interactive capture of calibration frames from the camera stream.

Shows detected ChArUco corners live, plus all corners captured so far (green),
so you can see which image regions still need coverage.

Keys: SPACE save frame, Q / ESC quit.
"""

import argparse
from pathlib import Path

import cv2

from calib.board import BoardConfig, detect, effective_min_corners, make_detector
from calib.intrinsics import coverage
from calib.io import load_yaml
from record.stream import capture


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--board", default="config/board.yaml")
    p.add_argument("--source", help="override stream URL / device index / video file")
    p.add_argument("-o", "--out", default="data/intrinsics/frames")
    a = p.parse_args()

    cam = load_yaml(a.camera)
    min_corners = cam.get("intrinsics", {}).get("min_corners_per_view", 8)

    board = BoardConfig.from_yaml(a.board).build()
    detector = make_detector(board)
    min_corners = effective_min_corners(board, min_corners)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = len(list(out.glob("*.png")))


    history = []  # detections of saved frames
    with capture(cam, a.source) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("stream ended or dropped")
                break
            det = detect(detector, frame)
            view = frame.copy()
            for d in history:
                for x, y in d.corners.reshape(-1, 2):
                    cv2.circle(view, (int(x), int(y)), 3, (0, 200, 0), -1, cv2.LINE_AA)
            n = 0 if det is None else len(det)
            if det is not None:
                cv2.aruco.drawDetectedCornersCharuco(view, det.corners, det.ids, (0, 0, 255))
            size = (frame.shape[1], frame.shape[0])
            cov = coverage(history, size) if history else 0.0
            color = (0, 255, 0) if n >= min_corners else (0, 0, 255)
            cv2.putText(view, f"corners {n}  saved {saved}  coverage {cov:.0%}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            scale = min(1.0, 1280 / frame.shape[1])
            cv2.imshow("capture", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if n < min_corners:
                    print(f"skipped: only {n} corners (< {min_corners})")
                    continue
                path = out / f"frame_{saved:04d}.png"
                cv2.imwrite(str(path), frame)
                history.append(det)
                saved += 1
                print(f"saved {path} ({n} corners)")


if __name__ == "__main__":
    main()
