"""Solve the Vicon world -> camera extrinsics from a static pose session.

Prints per-pose errors and writes data/extrinsics/extrinsics_<date>.yaml, which also
contains the intrinsics it was solved with (one file describes the whole camera).
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from calib.board import BoardConfig
from calib.extrinsics import calibrate
from calib.io import load_intrinsics, load_static_session, load_yaml, save_extrinsics
from calib.transforms import inv_T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session", help="session folder from capture_static_poses.py")
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--intrinsics", help="override the intrinsics recorded in the session")
    p.add_argument("-o", "--out", help="output YAML (default data/extrinsics/extrinsics_<date>.yaml)")
    a = p.parse_args()

    meta, poses = load_static_session(a.session)
    board_cfg = BoardConfig(**meta["board"])
    intr_path = Path(a.intrinsics or meta["intrinsics_file"])
    model, size, K, dist = load_intrinsics(intr_path)
    print(f"{len(poses)} poses from {a.session}, intrinsics {intr_path} ({model})")

    res = calibrate(board_cfg.build(), poses, model, K, dist)

    for w in res.warnings:
        print(f"WARNING: {w}")
    print(f"\nreprojection RMS {res.rms_px:.3f} px (initial guess {res.init_rms_px:.2f} px)")
    print("pose       rms px   single-image PnP diff: mm   deg  (mostly PnP depth noise)")
    for name, rms in res.per_pose_rms_px.items():
        t_mm, r_deg = res.per_pose_err[name]
        print(f"{name}  {rms:7.3f}   {t_mm:26.1f} {r_deg:5.2f}")
    if res.rejected:
        print(f"rejected outlier poses: {', '.join(res.rejected)} (check them: board moved, marker swapped?)")

    # Where in the image are the residuals large? 3x3 grid summary.
    xy, r = res.residuals[:, :2], np.linalg.norm(res.residuals[:, 2:], axis=1)
    gx = np.clip((xy[:, 0] / size[0] * 3).astype(int), 0, 2)
    gy = np.clip((xy[:, 1] / size[1] * 3).astype(int), 0, 2)
    print("\nRMS per image region [px] (- = no data, capture poses there):")
    for y in range(3):
        cells = []
        for x in range(3):
            m = (gx == x) & (gy == y)
            cells.append(f"{np.sqrt(np.mean(r[m] ** 2)):6.2f}" if m.any() else "     -")
        print("   " + " ".join(cells))

    T_wc = inv_T(res.T_cam_world)
    print(f"\ncamera position in Vicon world: {np.round(T_wc[:3, 3], 3).tolist()} m")
    print(f"camera viewing direction (world): {np.round(T_wc[:3, 2], 3).tolist()}")

    out = a.out or f"data/extrinsics/extrinsics_{date.today()}.yaml"
    save_extrinsics(out, res, intr_path, load_yaml(a.camera), meta["board"], a.session)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
