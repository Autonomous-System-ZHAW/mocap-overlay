"""Measure the camera delay from a recording of a moving ChArUco board (tracked by Vicon).

Record ~30 s while moving the board at moderate speed in front of the camera:
    uv run python scripts/record_session.py --duration 30 --note "delay board"
then:
    uv run python scripts/estimate_time_offset.py [data/recordings/<ts>]

Writes data/time_sync/time_sync_<date>.yaml (and a copy into the recording) plus a
comparison image (red: corners predicted without delay correction, green: with).
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from calib.board import BoardConfig, Detection, detect, make_detector
from calib.camera import Camera
from calib.io import load_yaml
from calib.time_sync import Problem, collect_corners, estimate, frame_times
from calib.transforms import apply
from record.recording import Recording


def newest_board_recording(subject: str) -> Path:
    for d in sorted(Path("data/recordings").glob("*/"), reverse=True):
        try:
            r = Recording(d)
        except (OSError, ValueError):
            continue
        if subject in r.tracks and (~r.tracks[subject].occluded).sum() > 100:
            return d
    raise SystemExit(f"no recording with Vicon subject {subject!r} found in data/recordings")


def load_detections(rec: Recording, board, min_corners: int) -> dict[int, Detection]:
    """Detect the board in every frame; cached in the recording folder."""
    cache = rec.path / "board_detections.npz"
    if cache.exists():
        d = np.load(cache)
        out = {}
        for i in np.unique(d["frame"]):
            m = d["frame"] == i
            out[int(i)] = Detection(d["corners"][m].reshape(-1, 1, 2), d["ids"][m].reshape(-1, 1))
        return out
    detector, out = make_detector(board), {}
    for i, _, img in rec.iter_frames():
        det = detect(detector, img)
        if det is not None and len(det) >= min_corners:
            out[i] = det
        if i % 100 == 0:
            print(f"  detecting board: frame {i}/{len(rec.frames)}", end="\r")
    print()
    frames = np.concatenate([np.full(len(v), k) for k, v in out.items()]) if out else np.zeros(0, int)
    np.savez(cache, frame=frames,
             corners=np.concatenate([v.corners.reshape(-1, 2) for v in out.values()]) if out else np.zeros((0, 2)),
             ids=np.concatenate([v.ids.ravel() for v in out.values()]) if out else np.zeros(0, int))
    return out


def comparison_image(rec: Recording, problem: Problem, best, out_path: Path) -> None:
    """Frame with the fastest board motion: predicted corners without (red) and with (green) correction."""
    e0 = problem.errors(0.0, 0.0)
    frame_idx = int(problem.c.frame[np.nanargmax(np.where(np.isfinite(e0), e0, -1))])
    img = next(im for i, _, im in rec.iter_frames(start=frame_idx) if i == frame_idx).copy()
    m = problem.c.frame == frame_idx
    for (d, r), color in (((0.0, 0.0), (0, 0, 255)), ((best.delay_s, best.readout_s), (0, 255, 0))):
        T_wb, ok = rec.pose_at(problem.subject, problem.t_frame[m] - d + problem.row[m] * r)
        for k in np.flatnonzero(ok):
            p_world = apply(T_wb[k] @ problem.T_body_board, problem.c.obj[m][k][None])
            px, v = problem.cam.project(p_world)
            if v[0]:
                cv2.drawMarker(img, tuple(np.round(px[0]).astype(int)), color, cv2.MARKER_CROSS, 40, 3, cv2.LINE_AA)
    for x, y in problem.c.img[m]:
        cv2.circle(img, (int(x), int(y)), 10, (255, 255, 0), 2, cv2.LINE_AA)
    text = (f"frame {frame_idx}: detected (cyan), predicted without delay (red), "
            f"with delay {best.delay_s * 1000:.0f} ms (green)")
    cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", nargs="?", help="recording folder (default: newest with the board tracked)")
    p.add_argument("--vicon", default="config/vicon.yaml")
    p.add_argument("--calibration", help="extrinsics YAML (default: the copy in the recording)")
    p.add_argument("--min-corners", type=int, default=4)
    a = p.parse_args()

    subject = load_yaml(a.vicon)["subjects"]["board"]
    rec = Recording(a.recording or newest_board_recording(subject))
    calib_path = Path(a.calibration) if a.calibration else rec.calibration_path
    if calib_path is None:
        raise SystemExit("recording has no calibration.yaml, pass --calibration")
    calib = load_yaml(calib_path)
    cam = Camera.from_extrinsics_yaml(calib_path)
    board = BoardConfig(**calib["board"]).build()
    print(f"recording {rec.path} ({rec.meta.get('note', '')}), calibration {calib_path}")

    dets = load_detections(rec, board, a.min_corners)
    corners = collect_corners(dets, board)
    print(f"board detected in {len(dets)}/{len(rec.frames)} frames, {len(corners.frame)} corners")
    if len(dets) < 50:
        raise SystemExit("too few frames with the board, record again with the board in view")

    results = []
    for timebase in ("receive", "pts"):
        try:
            prob = Problem(rec, subject, cam, np.array(calib["T_body_board"]), corners, frame_times(rec, timebase))
        except ValueError as e:
            print(f"[{timebase}] skipped: {e}")
            continue
        res = estimate(prob, timebase)
        results.append((res, prob))
        spread = np.std([d for _, d in res.windows]) * 1000 if len(res.windows) > 1 else float("nan")
        print(f"\n[{timebase}] delay {res.delay_s * 1000:.1f} ms, rolling-shutter readout {res.readout_s * 1000:.0f} ms")
        print(f"  reprojection RMS {res.rms_px:.2f} px (without correction {res.rms_px_no_delay:.1f} px), "
              f"inliers {res.inlier_ratio:.0%}")
        print(f"  per {5:.0f} s window: " + ", ".join(f"{d * 1000:.0f}" for _, d in res.windows)
              + f" ms  (std {spread:.1f} ms)")

    best, best_prob = min(results, key=lambda x: x[0].rms_px)
    xs, ys = best.curve
    print(f"\ncost curve ({best.timebase}):")
    for x, y in zip(xs[::10], ys[::10]):
        print(f"  {x * 1000:6.0f} ms {y:5.1f} px " + "#" * int(y * 1.5))

    out = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recording": str(rec.path),
        "camera_source": rec.meta.get("camera_source"),
        "camera_settings": rec.meta.get("camera_settings"),
        "timebase": best.timebase,
        "delay_s": round(best.delay_s, 4),
        "readout_s": round(best.readout_s, 4),
        "rms_px": round(best.rms_px, 3),
        "rms_px_no_delay": round(best.rms_px_no_delay, 3),
        "window_delays_s": [round(d, 4) for _, d in best.windows],
        "all": {r.timebase: {"delay_s": round(r.delay_s, 4), "readout_s": round(r.readout_s, 4),
                             "rms_px": round(r.rms_px, 3)} for r, _ in results},
        "usage": "exposure time of a frame = frame time (timebase) - delay_s; row y at + (y/H - 0.5) * readout_s",
    }
    path = Path("data/time_sync") / f"time_sync_{date.today()}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (path, rec.path / "time_sync.yaml"):
        with open(target, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
    img_path = rec.path / "time_sync_check.jpg"
    comparison_image(rec, best_prob, best, img_path)
    print(f"\nbest: {best.timebase}, delay {best.delay_s * 1000:.1f} ms -> saved {path}\ncheck image: {img_path}")


if __name__ == "__main__":
    main()
