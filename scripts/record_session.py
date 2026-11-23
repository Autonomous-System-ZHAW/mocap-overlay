"""Record camera video + Vicon poses (+ markers) into data/recordings/<timestamp>/.

All streams are timestamped on this machine's clock. The camera delay is not removed
here; it is measured afterwards (time sync) and applied when the recording is used.

Keys (preview window): Q / ESC stop.  Headless (--no-preview): Ctrl+C stops.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from calib.io import load_yaml, load_vicon_config
from record.camera_stream import CameraStream
from record.recorder import Recorder, RecState
from record.vicon_stream import ViconStream


def newest(folder: str, pattern: str) -> Path | None:
    files = sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def status_line(rec: Recorder, cam: CameraStream, vicon: ViconStream | None) -> str:
    s = rec.stats()
    line = (f"REC {s['duration_s']:6.1f} s  frames {s['frames']} (dropped {s['frames_dropped']}, "
            f"{cam.fps:4.1f} fps)")
    if vicon is not None:
        line += f"  vicon {vicon.state.value} {s['vicon_samples']} samples (dropped {s['vicon_dropped']})"
    return line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="config/camera.yaml")
    p.add_argument("--vicon", default="config/vicon.yaml")
    p.add_argument("--source", help="override stream URL / device index / video file")
    p.add_argument("--no-vicon", action="store_true", help="record video only")
    p.add_argument("--subjects", nargs="+", help="Vicon subjects (default: all in vicon.yaml)")
    p.add_argument("--markers", action="store_true", help="also record all marker positions")
    p.add_argument("--calibration", help="extrinsics YAML to store with the recording (default: newest)")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--codec", choices=["x264", "nvenc"], default="x264")
    p.add_argument("--quality", type=int, default=18, help="x264 CRF / NVENC CQ, lower = better")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--note", default="", help="free text stored in meta.yaml")
    p.add_argument("-o", "--out", help="output folder (default data/recordings/<timestamp>)")
    a = p.parse_args()

    cam_cfg = load_yaml(a.camera)
    out = Path(a.out or Path("data/recordings") / datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    calibration = a.calibration or newest("data/extrinsics", "extrinsics_*.yaml")

    vicon = None
    if not a.no_vicon:
        vcfg = load_vicon_config(a.vicon)
        subjects = a.subjects or list(dict.fromkeys(vcfg["subjects"].values()))
        vicon = ViconStream(vcfg["host"], vcfg["sdk_dir"], subjects, vcfg.get("stream_mode", "ServerPush"),
                            with_markers=a.markers)

    camera = CameraStream(cam_cfg, a.source)
    fps = cam_cfg.get("settings", {}).get("fps", 25)
    rec = Recorder(out, camera, vicon, a.codec, a.quality, fps, a.markers,
                   meta={"note": a.note, "camera_name": cam_cfg.get("name"),
                         "camera_settings": cam_cfg.get("settings", {})},
                   calibration=calibration)

    with camera:
        if vicon is not None:
            vicon.start()
        try:
            print("waiting for camera" + (" and Vicon" if vicon else "") + " ...")
            if camera.read(timeout=30) is None:
                raise SystemExit("no camera frames")
            if vicon is not None and not vicon.wait_streaming(timeout=30):
                raise SystemExit(f"Vicon not streaming: {vicon.last_error}")
            rec.start()
            print(f"recording to {out}")
            t_print = 0.0
            while rec.state == RecState.RECORDING:
                if a.duration and time.time() - rec.t_start >= a.duration:
                    break
                if time.monotonic() - t_print > 2:
                    print(status_line(rec, camera, vicon))
                    t_print = time.monotonic()
                if a.no_preview:
                    time.sleep(0.1)
                    continue
                frame = camera.read(timeout=1)
                if frame is None:
                    continue
                view = cv2.resize(frame.image, None, fx=0.5, fy=0.5)
                cv2.circle(view, (25, 25), 10, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(view, status_line(rec, camera, vicon), (45, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("recording", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            rec.stop()  # also on Ctrl+C: drains the queues and closes the files
            if vicon is not None:
                vicon.stop()
            if rec.state != RecState.IDLE:
                print(f"{rec.state.value}: {rec.stats()}\n-> {out}")


if __name__ == "__main__":
    main()
