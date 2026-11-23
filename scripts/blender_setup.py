"""Open a recording in Blender with the calibrated camera and the Vicon animation.

    uv run python scripts/blender_setup.py data/recordings/<ts>

First run: prepares data/blender/<recording>/ (camera, delay-corrected poses, undistorted
camera video as background, lens map for the compositor), builds mocap_scene.blend and opens
it. Later runs just open the existing .blend: your modelling is never overwritten
(--rebuild creates a fresh file; the old one is kept as a backup).

In Blender: Numpad 0 = camera view over the video, play the timeline to see the Vicon motion.
See the README_mocap text block inside the .blend for the structure.
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calib.camera import Camera
from calib.time_sync import load_delay
from calib.virtual_camera import VirtualCamera
from record.recording import Recording
from record.writers import FFmpegVideoWriter

SETUP_SCRIPT = Path(__file__).resolve().parent.parent / "blender" / "setup_scene.py"
LIGHT_DIR = [0.3, 0.2, 1.0]  # same as the live renderer
OPEN_IN_CAMERA_VIEW = """
import bpy
def _camera_view():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces[0].region_3d.view_perspective = "CAMERA"
                area.spaces[0].shading.type = "MATERIAL"
    return None
bpy.app.timers.register(_camera_view, first_interval=0.5)
"""


def write_background(rec: Recording, cam: Camera, vcam: VirtualCamera, path: Path, scale: float) -> None:
    """Camera video resampled into the virtual pinhole image (what the Blender camera sees)."""
    mx, my, size = vcam.undistort_maps(cam, scale)
    fps = float(1 / np.median(np.diff(rec.frame_times)))
    writer = FFmpegVideoWriter(path, size, fps, quality=20, extra_args=["-g", "5"])  # short GOP: smooth scrubbing
    try:
        for i, _, img in rec.iter_frames():
            writer.write(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
            if i % 100 == 0:
                print(f"  background video {i}/{len(rec.frames)}", end="\r")
    finally:
        writer.close()
    print()


def prepare(rec: Recording, calib: Path, out: Path, delay: float, bg_scale: float) -> Path:
    cam = Camera.from_extrinsics_yaml(calib)
    vcam = VirtualCamera.from_camera(cam)
    out.mkdir(parents=True, exist_ok=True)
    t_exposure = rec.frame_times - delay

    poses = {}
    for subject in sorted(rec.tracks):
        T, ok = rec.pose_at(subject, t_exposure)
        T[~ok] = np.nan
        poses[subject] = T
        print(f"  {subject}: tracked in {ok.mean():.0%} of the frames")
    np.savez(out / "poses.npz", **poses)

    # Lens map for the compositor's Map UV node: real pixel -> UV in the render (v = 0 at the bottom).
    w, h = vcam.size
    uv = np.stack([(vcam.map_x + 0.5) / w, 1 - (vcam.map_y + 0.5) / h], axis=-1).astype(np.float32)
    uv[vcam.map_x < 0] = -1.0
    np.save(out / "distort_uv.npy", uv)

    print(f"virtual camera {w}x{h}, writing undistorted background video ...")
    write_background(rec, cam, vcam, out / "background.mp4", bg_scale)

    cfg = {
        "recording": str(rec.path.resolve()),
        "calibration": str(calib.resolve()),
        "delay_s": delay,
        "fps": float(1 / np.median(np.diff(rec.frame_times))),
        "frame_count": len(rec.frames),
        "image_size": list(cam.image_size),
        "virtual": {"K": vcam.K.tolist(), "size": list(vcam.size)},
        "T_world_cam": cam.T_world_cam.tolist(),
        "subjects": sorted(rec.tracks),
        "light_dir": LIGHT_DIR,
        "files": {"background": "background.mp4", "poses": "poses.npz", "distort_uv": "distort_uv.npy",
                  "video": str(rec.video_path.resolve())},
    }
    path = out / "setup.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--calibration", help="extrinsics YAML (default: the copy in the recording)")
    p.add_argument("--time-sync", help="time sync YAML (default: newest in data/time_sync)")
    p.add_argument("--delay", type=float, help="camera delay [s], overrides the time sync file")
    p.add_argument("--background-scale", type=float, default=0.5, help="resolution of the background video")
    p.add_argument("--rebuild", action="store_true", help="build a fresh .blend (the old one is backed up)")
    p.add_argument("--no-gui", action="store_true", help="only build the .blend, do not open Blender")
    p.add_argument("--blender", default="blender", help="Blender executable")
    p.add_argument("-o", "--out", help="folder (default data/blender/<recording>)")
    a = p.parse_args()

    if shutil.which(a.blender) is None:
        raise SystemExit(f"Blender not found ({a.blender}); pass --blender /path/to/blender")
    rec = Recording(a.recording)
    out = Path(a.out or Path("data/blender") / rec.path.name)
    blend = out / "mocap_scene.blend"

    if blend.exists() and a.rebuild:
        backup = blend.with_name(f"mocap_scene_backup_{datetime.now():%Y%m%d_%H%M%S}.blend")
        blend.rename(backup)
        print(f"old scene kept as {backup}")
    if not blend.exists():
        calib = Path(a.calibration) if a.calibration else rec.calibration_path
        if calib is None:
            raise SystemExit("recording has no calibration.yaml, pass --calibration")
        delay, _, sync_path = load_delay(a.time_sync)
        if a.delay is not None:
            delay, sync_path = a.delay, "--delay"
        print(f"recording {rec.path} ({rec.meta.get('note', '')}), calibration {calib}, "
              f"delay {delay * 1000:.0f} ms ({sync_path or 'not measured!'})")
        setup = prepare(rec, calib, out, delay, a.background_scale)
        subprocess.run([a.blender, "-b", "--factory-startup", "--python", str(SETUP_SCRIPT), "--",
                        str(setup), str(blend)], check=True)
    else:
        print(f"opening existing {blend} (use --rebuild for a fresh scene)")

    if not a.no_gui:
        subprocess.Popen([a.blender, str(blend), "--python-expr", OPEN_IN_CAMERA_VIEW])


if __name__ == "__main__":
    main()
