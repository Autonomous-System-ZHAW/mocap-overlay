"""Calibrate camera intrinsics from a folder of ChArUco images.

Writes a YAML result (with date and camera settings) and prints per-view errors.
Use --compare to run all three lens models and pick the best one.
"""

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

import cv2

from calib.board import BoardConfig, detect, effective_min_corners, make_detector
from calib.intrinsics import MODELS, calibrate
from calib.io import load_yaml, save_intrinsics

TARGET_RMS_PX = 0.5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--board", default="config/board.yaml")
    p.add_argument("--frames", default="data/intrinsics/frames")
    p.add_argument("--model", choices=MODELS, help="override the model from camera.yaml")
    p.add_argument("--compare", action="store_true", help="calibrate with all models and report")
    p.add_argument("-o", "--out", help="output YAML (default data/intrinsics/intrinsics_<date>_<model>.yaml)")
    a = p.parse_args()

    cam = load_yaml(a.camera)
    icfg = cam.get("intrinsics", {})
    board_cfg = BoardConfig.from_yaml(a.board)
    board = board_cfg.build()
    detector = make_detector(board)

    files = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise SystemExit(f"no images in {a.frames}")

    detections, image_size = {}, None
    for f in files:
        img = cv2.imread(str(f))
        size = (img.shape[1], img.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise SystemExit(f"{f.name}: size {size} differs from {image_size}; mixed stream profiles?")
        det = detect(detector, img)
        if det is not None:
            detections[f.name] = det
    print(f"{len(detections)}/{len(files)} images with detections, image size {image_size}")
    expected = cam.get("settings", {}).get("resolution")
    if expected and tuple(expected) != image_size:
        print(f"WARNING: images are {image_size} but camera.yaml says {tuple(expected)}; "
              "the saved camera settings will not match this calibration")

    models = MODELS if a.compare else [a.model or icfg.get("model", "pinhole")]
    for model in models:
        try:
            res = calibrate(
                board, detections, image_size, model=model,
                min_corners=effective_min_corners(board, icfg.get("min_corners_per_view", 8)),
                outlier_factor=icfg.get("outlier_factor", 3.0),
            )
        except cv2.error as e:
            print(f"[{model}] failed: {e.err}")
            continue

        status = "OK" if res.rms_px < TARGET_RMS_PX else f"ABOVE TARGET {TARGET_RMS_PX}"
        print(f"\n[{model}] RMS {res.rms_px:.3f} px ({status}), {len(res.view_names)} views, "
              f"coverage {res.coverage:.0%}")
        print(f"  fx {res.K[0, 0]:.1f} fy {res.K[1, 1]:.1f} cx {res.K[0, 2]:.1f} cy {res.K[1, 2]:.1f}")
        print(f"  dist {res.dist.round(5).tolist()}")
        if res.rejected_views:
            print(f"  rejected outliers: {', '.join(res.rejected_views)}")
        worst = sorted(zip(res.per_view_rms_px, res.view_names), reverse=True)[:5]
        print("  worst views: " + ", ".join(f"{n} {e:.2f}" for e, n in worst))
        if res.coverage < 0.75:
            print("  hint: low coverage, capture more views with the board near the image corners/edges")

        out = a.out if a.out and not a.compare else f"data/intrinsics/intrinsics_{date.today()}_{model}.yaml"
        save_intrinsics(out, res, cam, asdict(board_cfg))
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
