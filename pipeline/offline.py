"""Offline overlay: recording (video + Vicon log) + scene -> rendered video.

Every video frame is rendered with the Vicon poses at its exposure time
(arrival time - camera delay), so virtual objects stay on the moving real ones.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calib.camera import Camera
from calib.transforms import apply
from overlay import draw
from overlay.gl_renderer import GLRenderer, composite
from overlay.scene import Scene
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


def trail_points(rec: Recording, subject: str, t: float, seconds: float, offset: np.ndarray,
                 rate_hz: float = 50.0) -> list[np.ndarray]:
    """World positions of `offset` (subject frame) over the last `seconds`, split at gaps."""
    times = np.linspace(t - seconds, t, max(2, int(seconds * rate_hz)))
    T, ok = rec.pose_at(subject, times)
    segments, cur = [], []
    for Ti, v in zip(T, ok):
        if v:
            cur.append(apply(Ti, offset[None])[0])
        elif len(cur) > 1:
            segments.append(np.array(cur))
            cur = []
        else:
            cur = []
    if len(cur) > 1:
        segments.append(np.array(cur))
    return segments


def render_recording(rec: Recording, scene: Scene, cam: Camera, out_path: str | Path, opt: OfflineOptions) -> dict:
    t_frames = rec.frame_times
    fps = float(1 / np.median(np.diff(t_frames)))
    first = int(np.searchsorted(t_frames, t_frames[0] + opt.start_s))
    last = len(t_frames) if opt.duration_s is None else int(np.searchsorted(t_frames, t_frames[first] + opt.duration_s))
    missing = scene.subjects - set(rec.tracks)
    if missing:
        print(f"WARNING: subjects not in the recording, their objects are skipped: {sorted(missing)}")

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
            t = t_host - opt.delay_s  # exposure time on the host clock
            poses = {}
            for s in scene.subjects & set(rec.tracks):
                T, ok = rec.pose_at(s, [t])
                poses[s] = T[0] if ok[0] else None
            items, holdouts = scene.draw_items(poses)
            out = composite(img, renderer.render(items, holdouts))
            for tr in scene.trails:
                if tr.subject in rec.tracks:
                    for seg in trail_points(rec, tr.subject, t, tr.seconds, tr.offset):
                        draw.polyline(out, cam, seg, tr.color_bgr, tr.width, step_m=0.02)
            if opt.label:
                cv2.putText(out, opt.label, (30, cam.image_size[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 255), 2, cv2.LINE_AA)
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
