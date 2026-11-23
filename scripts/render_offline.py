"""Render an overlay scene onto a recording.

    uv run python scripts/render_offline.py data/recordings/<ts> --scene config/scenes/demo_street.yaml

Uses the calibration stored in the recording and the newest camera delay
(data/time_sync), writes data/renders/<recording>_<scene>.mp4.
"""

import argparse
from pathlib import Path

from calib.camera import Camera
from calib.time_sync import load_delay
from overlay.scene import Scene
from pipeline.offline import OfflineOptions, render_recording
from record.recording import Recording


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--scene", default="config/scenes/demo_street.yaml")
    p.add_argument("--calibration", help="extrinsics YAML (default: the copy in the recording)")
    p.add_argument("--time-sync", help="time sync YAML (default: newest in data/time_sync)")
    p.add_argument("--delay", type=float, help="camera delay [s], overrides the time sync file")
    p.add_argument("--start", type=float, default=0.0, help="start offset [s]")
    p.add_argument("--duration", type=float, help="render only this many seconds")
    p.add_argument("--codec", choices=["x264", "nvenc"], default="x264")
    p.add_argument("--quality", type=int, default=18)
    p.add_argument("--render-scale", type=float, default=1.0, help="virtual render resolution factor")
    p.add_argument("-o", "--out", help="output .mp4 (default data/renders/<recording>_<scene>.mp4)")
    a = p.parse_args()

    rec = Recording(a.recording)
    calib = Path(a.calibration) if a.calibration else rec.calibration_path
    if calib is None:
        raise SystemExit("recording has no calibration.yaml, pass --calibration")
    delay, _, sync_path = load_delay(a.time_sync)
    if a.delay is not None:
        delay, sync_path = a.delay, "--delay"
    scene = Scene.from_yaml(a.scene)
    out = Path(a.out or f"data/renders/{rec.path.name}_{Path(a.scene).stem}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"recording {rec.path} ({rec.meta.get('note', '')}), scene {scene.name!r}")
    print(f"calibration {calib}, camera delay {delay * 1000:.0f} ms ({sync_path or 'not measured!'})")

    opt = OfflineOptions(delay_s=delay, start_s=a.start, duration_s=a.duration, codec=a.codec, quality=a.quality,
                         render_scale=a.render_scale, label=scene.name)
    render_recording(rec, scene, Camera.from_extrinsics_yaml(calib), out, opt)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
