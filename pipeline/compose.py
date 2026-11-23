"""One overlay frame: poses at the exposure time -> 3D render -> composite -> trails.
Shared by the offline renderer and the live view, so both look the same."""

import cv2
import numpy as np

from calib.camera import Camera
from overlay import draw
from overlay.gl_renderer import GLRenderer, composite
from overlay.scene import Scene
from pipeline.pose_source import PoseSource


def compose_frame(img: np.ndarray, t_exposure: float, scene: Scene, cam: Camera, renderer: GLRenderer,
                  poses: PoseSource, label: str | None = None) -> np.ndarray:
    current = {s: poses.pose_at(s, t_exposure) for s in scene.subjects}
    items, holdouts = scene.draw_items(current)
    out = composite(img, renderer.render(items, holdouts))
    for tr in scene.trails:
        for seg in poses.trail(tr.subject, t_exposure, tr.seconds, tr.offset):
            draw.polyline(out, cam, seg, tr.color_bgr, tr.width, step_m=0.02)
    if label:
        cv2.putText(out, label, (30, cam.image_size[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out
