"""Record camera video + Vicon poses with timestamps on the host clock.

State machine:  idle -> recording -> stopping -> done   (any -> error)

Recording folder:
    meta.yaml          what, when, which settings; stats are added on stop
    video.mkv          H.264, every frame the grabber delivered (row k of frames.csv = frame k)
    frames.csv         t_host, frame, stream_index, pts_ms
    vicon_poses.csv    t_host, t_recv, latency_s, frame_number, subject, tx, ty, tz, qx, qy, qz, qw, occluded
    markers.csv        t_host, frame_number, x, y, z           (optional)
    calibration.yaml   copy of the extrinsics used for the overlay (if available)

Each stream is written by its own thread from a subscription queue, so slow disk or
encoding never blocks the grabbers; if a writer falls behind, dropped items are counted.
"""

import shutil
import subprocess
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from record.camera_stream import CameraStream
from record.stream import redact
from record.subscription import Subscription
from record.vicon_stream import ViconSample, ViconStream
from record.writers import CsvLog, FFmpegVideoWriter

POSE_HEADER = ["t_host", "t_recv", "latency_s", "frame_number", "subject",
               "tx", "ty", "tz", "qx", "qy", "qz", "qw", "occluded"]


class RecState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPING = "stopping"
    DONE = "done"
    ERROR = "error"


def ntp_synchronized() -> bool | None:
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() == "yes"
    except (OSError, subprocess.TimeoutExpired):
        return None


def pose_rows(sample: ViconSample) -> list[list]:
    rows = []
    for subject, T in sample.poses.items():
        head = [f"{sample.t_host:.6f}", f"{sample.t_recv:.6f}", f"{sample.latency_s:.6f}", sample.frame_number, subject]
        if T is None:
            rows.append(head + [""] * 7 + [1])
        else:
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
            rows.append(head + [f"{v:.6f}" for v in T[:3, 3]] + [f"{v:.8f}" for v in q] + [0])
    return rows


class Recorder:
    def __init__(self, out_dir: str | Path, camera: CameraStream, vicon: ViconStream | None = None,
                 codec: str = "x264", quality: int = 18, fps: float = 25.0, record_markers: bool = False,
                 meta: dict | None = None, calibration: str | Path | None = None):
        self.out = Path(out_dir)
        self.camera, self.vicon = camera, vicon
        self.codec, self.quality, self.fps = codec, quality, fps
        self.record_markers = record_markers
        self.meta = dict(meta or {})
        self.calibration = Path(calibration) if calibration else None
        self.state = RecState.IDLE
        self.error = ""
        self.frames_written = 0
        self.vicon_samples = 0
        self.t_start = self.t_stop = 0.0
        self._cam_sub: Subscription | None = None
        self._vicon_sub: Subscription | None = None
        self._threads: list[threading.Thread] = []

    # --- control ------------------------------------------------------------------
    def start(self) -> None:
        if self.state != RecState.IDLE:
            raise RuntimeError(f"cannot start in state {self.state.value}")
        self.out.mkdir(parents=True, exist_ok=False)
        if self.calibration and self.calibration.exists():
            shutil.copy(self.calibration, self.out / "calibration.yaml")
        self.t_start = time.time()
        self.meta.update({
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "t_start": self.t_start,
            "camera_source": redact(self.camera.source),
            "ntp_synchronized": ntp_synchronized(),
            "video": {"file": "video.mkv", "codec": self.codec, "quality": self.quality, "nominal_fps": self.fps},
            "calibration": "calibration.yaml" if (self.out / "calibration.yaml").exists() else None,
        })
        if self.vicon is not None:
            self.meta["vicon"] = {"host": self.vicon.host, "subjects": self.vicon.subjects,
                                  "frame_rate": self.vicon.frame_rate}
        self._write_meta()

        self._cam_sub = self.camera.subscribe()
        self._threads = [threading.Thread(target=self._camera_writer, name="rec-camera", daemon=True)]
        if self.vicon is not None:
            self._vicon_sub = self.vicon.subscribe()
            self._threads.append(threading.Thread(target=self._vicon_writer, name="rec-vicon", daemon=True))
        self.state = RecState.RECORDING
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        if self.state not in (RecState.RECORDING, RecState.ERROR):
            return
        failed = self.state == RecState.ERROR
        self.state = RecState.STOPPING
        self.t_stop = time.time()
        self.camera.unsubscribe(self._cam_sub)
        self._cam_sub.put(None)  # sentinel: writer drains the queue, then closes the files
        if self._vicon_sub is not None:
            self.vicon.unsubscribe(self._vicon_sub)
            self._vicon_sub.put(None)
        for t in self._threads:
            t.join()
        self.meta["stats"] = self.stats()
        self._write_meta()
        self.state = RecState.ERROR if failed or self.error else RecState.DONE

    def stats(self) -> dict:
        duration = (self.t_stop or time.time()) - self.t_start if self.t_start else 0.0
        return {
            "duration_s": round(duration, 2),
            "frames": self.frames_written,
            "frames_dropped": self._cam_sub.dropped if self._cam_sub else 0,
            "vicon_samples": self.vicon_samples,
            "vicon_dropped": self._vicon_sub.dropped if self._vicon_sub else 0,
            "error": self.error or None,
        }

    def _write_meta(self) -> None:
        with open(self.out / "meta.yaml", "w") as f:
            yaml.safe_dump(self.meta, f, sort_keys=False, default_flow_style=None)

    def _fail(self, msg: str) -> None:
        print(f"[recorder] ERROR {msg}")
        self.error = msg
        self.state = RecState.ERROR

    # --- writer threads -----------------------------------------------------------
    def _camera_writer(self) -> None:
        video, log = None, CsvLog(self.out / "frames.csv", ["t_host", "frame", "stream_index", "pts_ms"])
        try:
            while (frame := self._cam_sub.get()) is not None:
                if video is None:
                    size = (frame.image.shape[1], frame.image.shape[0])
                    video = FFmpegVideoWriter(self.out / "video.mkv", size, self.fps, self.codec, self.quality)
                    self.meta["video"]["size"] = list(size)
                try:
                    video.write(frame.image)
                except (ValueError, BrokenPipeError) as e:
                    self._fail(f"video: {e}")
                    continue  # keep draining the queue until stop()
                log.write([f"{frame.t_host:.6f}", self.frames_written, frame.index, f"{frame.pts_ms:.3f}"])
                self.frames_written += 1
        finally:
            log.close()
            if video is not None and video.close() != 0:
                self._fail("ffmpeg exited with an error")

    def _vicon_writer(self) -> None:
        poses = CsvLog(self.out / "vicon_poses.csv", POSE_HEADER)
        markers = CsvLog(self.out / "markers.csv", ["t_host", "frame_number", "x", "y", "z"]) \
            if self.record_markers else None
        try:
            while (sample := self._vicon_sub.get()) is not None:
                for row in pose_rows(sample):
                    poses.write(row)
                if markers is not None and sample.markers is not None:
                    for p in np.asarray(sample.markers).reshape(-1, 3):
                        markers.write([f"{sample.t_host:.6f}", sample.frame_number, *(f"{v:.6f}" for v in p)])
                self.vicon_samples += 1
        finally:
            poses.close()
            if markers is not None:
                markers.close()
