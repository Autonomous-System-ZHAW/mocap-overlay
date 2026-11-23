"""Live 3D overlay on the camera stream.

    uv run python scripts/live_overlay.py --scene config/scenes/demo_street.yaml

Live: camera (config/camera.yaml) + Vicon (config/vicon.yaml), newest calibration in
data/extrinsics and newest camera delay in data/time_sync.

Replay a recording as if it were live (no lab needed, e.g. to build scenes):
    uv run python scripts/live_overlay.py --replay data/recordings/<ts>

Keys: Q quit, D delay correction, O overlay, H info, F fullscreen, S snapshot.
"""

import argparse
from pathlib import Path

from calib.camera import Camera
from calib.io import load_yaml
from calib.time_sync import load_delay
from overlay.scene import Scene
from pipeline.live import LiveOptions, LiveOverlay
from pipeline.pose_source import LiveViconSource, RecordingSource
from record.camera_stream import CameraStream
from record.recording import Recording
from record.vicon_stream import ViconStream


def newest(folder: str, pattern: str) -> Path | None:
    files = sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="config/scenes/demo_street.yaml")
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--vicon", default="config/vicon.yaml")
    p.add_argument("--calibration", help="extrinsics YAML (default: newest, or the recording's copy in replay)")
    p.add_argument("--time-sync", help="time sync YAML (default: newest in data/time_sync)")
    p.add_argument("--delay", type=float, help="camera delay [s], overrides the time sync file")
    p.add_argument("--replay", help="recording folder to play back instead of camera + Vicon")
    p.add_argument("--source", help="override the camera stream URL / device")
    p.add_argument("--render-scale", type=float, default=1.0, help="virtual render resolution (lower = faster)")
    p.add_argument("--width", type=int, default=1600, help="initial window width [px]")
    a = p.parse_args()

    scene = Scene.from_yaml(a.scene)
    delay, _, sync_path = load_delay(a.time_sync)
    if a.delay is not None:
        delay, sync_path = a.delay, "--delay"
    vicon = None

    if a.replay:
        rec = Recording(a.replay)
        calib = Path(a.calibration) if a.calibration else rec.calibration_path
        camera = CameraStream({"rtsp_url": ""}, str(rec.video_path))  # plays back in real time
        poses = RecordingSource(rec)
        times = rec.frame_times

        def frame_time(f):  # recorded arrival time of this video frame
            return float(times[min(f.index, len(times) - 1)])
        print(f"replay {rec.path} ({rec.meta.get('note', '')})")
    else:
        calib = Path(a.calibration) if a.calibration else newest("data/extrinsics", "extrinsics_*.yaml")
        camera = CameraStream(load_yaml(a.camera), a.source)
        vcfg = load_yaml(a.vicon)
        trail_s = max([t.seconds for t in scene.trails], default=0.0)
        vicon = ViconStream(vcfg["host"], vcfg["sdk_dir"], sorted(scene.subjects), vcfg.get("stream_mode", "ServerPush"),
                            buffer_s=max(10.0, delay + trail_s + 2.0))
        poses = LiveViconSource(vicon)

        def frame_time(f):
            return f.t_host

    if calib is None:
        raise SystemExit("no calibration found, pass --calibration")
    print(f"scene {scene.name!r} (subjects: {sorted(scene.subjects) or 'none'}), calibration {calib}")
    print(f"camera delay {delay * 1000:.0f} ms ({sync_path or 'not measured!'})")

    live = LiveOverlay(camera, poses, frame_time, scene, Camera.from_extrinsics_yaml(calib),
                       LiveOptions(delay_s=delay, render_scale=a.render_scale, window_width=a.width))
    with camera:
        if vicon is not None:
            vicon.start()
        try:
            live.run()
        finally:
            if vicon is not None:
                vicon.stop()
    print(f"{live.frames} frames shown")


if __name__ == "__main__":
    main()
