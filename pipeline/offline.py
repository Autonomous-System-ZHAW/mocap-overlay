"""Offline overlay: recording (video + Vicon log) + scene -> rendered video.

Every video frame is rendered with the Vicon poses at its exposure time
(arrival time - camera delay), so virtual objects stay on the moving real ones.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calib.camera import Camera
from overlay.gl_renderer import GLRenderer
from overlay.scene import Scene
from pipeline.compose import compose_frame
from pipeline.pose_source import RecordingSource
from record.recording import Recording
from record.writers import FFmpegVideoWriter


@dataclass
class OfflineOptions:
    delay_s: float = 0.0
    start_s: float = 0.0
    duration_s: float | None = None
    codec: str = "x264"
    quality: int = 18
    render_scale: float = 1.0
    label: str | None = None  # small text in the corner, e.g. the scene name


def render_recording(rec: Recording, scene: Scene, cam: Camera, out_path: str | Path, opt: OfflineOptions) -> dict:
    t_frames = rec.frame_times
    fps = float(1 / np.median(np.diff(t_frames)))
    first = int(np.searchsorted(t_frames, t_frames[0] + opt.start_s))
    last = len(t_frames) if opt.duration_s is None else int(np.searchsorted(t_frames, t_frames[first] + opt.duration_s))
    missing = scene.subjects - set(rec.tracks)
    if missing:
        print(f"WARNING: subjects not in the recording, their objects are skipped: {sorted(missing)}")

    poses = RecordingSource(rec)
    renderer = GLRenderer(cam, render_scale=opt.render_scale)
    print(f"virtual render size {renderer.virtual_size[0]}x{renderer.virtual_size[1]} for "
          f"{cam.image_size[0]}x{cam.image_size[1]} output, {last - first} frames at {fps:.2f} fps")
    writer = FFmpegVideoWriter(out_path, cam.image_size, fps, opt.codec, opt.quality)
    t0, n = time.monotonic(), 0
    try:
        for i, t_host, img in rec.iter_frames(start=first):
            if i >= last:
                break
            if (img.shape[1], img.shape[0]) != cam.image_size:
                raise ValueError(f"video is {img.shape[1]}x{img.shape[0]}, calibration {cam.image_size}")
            out = compose_frame(img, t_host - opt.delay_s, scene, cam, renderer, poses, opt.label)
            writer.write(out)
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{last - first} frames, {n / (time.monotonic() - t0):.1f} fps", end="\r")
    finally:
        writer.close()
        renderer.release()
    elapsed = time.monotonic() - t0
    print(f"\nrendered {n} frames in {elapsed:.1f} s ({n / max(elapsed, 1e-9):.1f} fps)")
    return {"frames": n, "seconds": elapsed}
