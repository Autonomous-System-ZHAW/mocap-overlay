"""Live overlay view: camera stream + Vicon -> 3D overlay in a window.

State machine:
    starting -> running <-> no_poses -> stopped
      running:  camera frames arrive and the pose source is ok -> full overlay
      no_poses: camera ok but Vicon is not streaming -> video only, with a warning

Always the newest camera frame is shown; if rendering is slower than the camera,
frames are skipped instead of building up lag. Each frame is drawn with the poses at
its exposure time (frame time - camera delay).
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from calib.camera import Camera
from overlay.gl_renderer import GLRenderer
from overlay.scene import Scene
from pipeline.compose import compose_frame
from pipeline.pose_source import PoseSource
from record.camera_stream import CameraFrame, CameraStream
from record.camera_stream import State as CameraState

WINDOW = "mocap overlay"
KEYS_HELP = "Q quit  D delay  O overlay  H info  F fullscreen  S snapshot"


class LiveState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    NO_POSES = "no_poses"
    STOPPED = "stopped"


@dataclass
class LiveOptions:
    delay_s: float = 0.0
    render_scale: float = 1.0
    window_width: int = 1600
    show_window: bool = True
    max_frames: int | None = None  # stop after this many frames (tests)
    snapshot_dir: str = "data/live/snapshots"


class LiveOverlay:
    def __init__(self, camera: CameraStream, poses: PoseSource, frame_time: Callable[[CameraFrame], float],
                 scene: Scene, cam: Camera, opt: LiveOptions,
                 on_frame: Callable[[np.ndarray, float], None] | None = None):
        """frame_time: host time of a frame (live: arrival time; replay: the recorded arrival time)."""
        self.camera, self.poses, self.frame_time = camera, poses, frame_time
        self.scene, self.cam, self.opt, self.on_frame = scene, cam, opt, on_frame
        self.state = LiveState.STARTING
        self.use_delay, self.show_overlay, self.show_info, self.fullscreen = True, True, True, False
        self.frames = 0
        self._fps = 0.0
        self._render_ms = 0.0

    def _set(self, state: LiveState, why: str = "") -> None:
        if state != self.state:
            print(f"[live] {self.state.value} -> {state.value}" + (f": {why}" if why else ""))
            self.state = state

    def run(self) -> None:
        renderer = GLRenderer(self.cam, render_scale=self.opt.render_scale)
        if self.opt.show_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            w, h = self.cam.image_size
            cv2.resizeWindow(WINDOW, self.opt.window_width, int(self.opt.window_width * h / w))
        t_last = time.monotonic()
        try:
            while self.state != LiveState.STOPPED:
                grabbed = self.camera.read(timeout=1.0)
                if grabbed is None:
                    if self.camera.state == CameraState.STOPPED:
                        self._set(LiveState.STOPPED, f"camera: {self.camera.last_error}")
                    elif self.opt.show_window:
                        self._handle_key(cv2.waitKey(1) & 0xFF, None)
                    continue
                img = grabbed.image
                if (img.shape[1], img.shape[0]) != self.cam.image_size:
                    raise SystemExit(f"camera delivers {img.shape[1]}x{img.shape[0]}, calibration is "
                                     f"{self.cam.image_size[0]}x{self.cam.image_size[1]}")

                ok, status = self.poses.status()
                self._set(LiveState.RUNNING if ok else LiveState.NO_POSES, "" if ok else status)
                t_exposure = self.frame_time(grabbed) - (self.opt.delay_s if self.use_delay else 0.0)
                t0 = time.monotonic()
                out = (compose_frame(img, t_exposure, self.scene, self.cam, renderer, self.poses)
                       if self.show_overlay and ok else img.copy())
                self._render_ms = 0.9 * self._render_ms + 0.1 * (time.monotonic() - t0) * 1000

                now = time.monotonic()
                self._fps = 0.9 * self._fps + 0.1 / max(now - t_last, 1e-6)
                t_last = now
                self.frames += 1
                if self.on_frame:
                    self.on_frame(out, t_exposure)
                if self.opt.show_window:
                    view = out.copy()
                    if self.show_info:
                        self._draw_info(view, grabbed, status, ok)
                    cv2.imshow(WINDOW, view)
                    self._handle_key(cv2.waitKey(1) & 0xFF, out)
                if self.opt.max_frames and self.frames >= self.opt.max_frames:
                    self._set(LiveState.STOPPED, "max frames")
        finally:
            renderer.release()
            if self.opt.show_window:
                cv2.destroyWindow(WINDOW)

    def _draw_info(self, view: np.ndarray, grabbed: CameraFrame, status: str, ok: bool) -> None:
        delay = f"{self.opt.delay_s * 1000:.0f} ms" if self.use_delay else "OFF"
        lines = [
            (f"{self.scene.name}   {self.state.value}   {status}", (0, 255, 0) if ok else (0, 0, 255)),
            (f"view {self._fps:4.1f} fps  camera {self.camera.fps:4.1f} fps  render {self._render_ms:4.0f} ms  "
             f"delay {delay}  overlay {'on' if self.show_overlay else 'OFF'}", (255, 255, 255)),
            (KEYS_HELP, (200, 200, 200)),
        ]
        for i, (text, color) in enumerate(lines):
            y = 45 + 40 * i
            cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    def _handle_key(self, key: int, out: np.ndarray | None) -> None:
        if key in (ord("q"), 27):
            self._set(LiveState.STOPPED, "quit")
        elif key == ord("d"):
            self.use_delay = not self.use_delay
        elif key == ord("o"):
            self.show_overlay = not self.show_overlay
        elif key == ord("h"):
            self.show_info = not self.show_info
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("s") and out is not None:
            path = Path(self.opt.snapshot_dir) / f"live_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), out)
            print(f"saved {path}")
