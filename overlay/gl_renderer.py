"""Offscreen OpenGL renderer (EGL, no window) for overlays that match the real lens.

Renderers only do ideal pinhole projection, the real lens is strongly distorted. So:
1. render with a virtual pinhole camera whose image is large enough to cover the whole
   real field of view (same focal length in the centre, "overscan" at the borders),
2. remap that image onto the real, distorted image with a precomputed lookup table.

The output is premultiplied RGBA at the real camera resolution; composite it with
`composite()`. Objects are drawn with simple Lambert shading. Holdouts write depth only:
they stand in for real objects (e.g. the real car) and hide virtual objects behind them.
"""

from dataclasses import dataclass

import cv2
import moderngl
import numpy as np

from calib.camera import Camera
from calib.intrinsics import project_points, undistort_points
from overlay.meshes import Mesh

VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_world;
void main() {
    vec4 world = model * vec4(in_pos, 1.0);
    v_world = world.xyz;
    v_normal = mat3(model) * in_normal;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec4 color;        // straight (not premultiplied) RGBA
uniform vec3 light_dir;    // world, towards the light
uniform vec3 cam_pos;      // world
uniform float ambient;
uniform bool shade;
in vec3 v_normal;
in vec3 v_world;
out vec4 f_color;
void main() {
    vec3 rgb = color.rgb;
    if (shade) {
        vec3 n = normalize(v_normal);
        if (dot(n, cam_pos - v_world) < 0.0) n = -n;   // face the camera, independent of winding
        float diffuse = max(dot(n, normalize(light_dir)), 0.0);
        rgb *= ambient + (1.0 - ambient) * diffuse;
    }
    f_color = vec4(rgb * color.a, color.a);            // premultiplied
}
"""


@dataclass
class DrawItem:
    mesh: Mesh
    T_world_obj: np.ndarray  # (4, 4)
    color: tuple[float, float, float, float]  # straight RGBA, 0..1
    shadow: bool = False  # cast a planar shadow onto the floor (z = 0)
    shade: bool = True


def projection_from_K(K: np.ndarray, size: tuple[int, int], near: float, far: float) -> np.ndarray:
    """OpenGL projection for an OpenCV camera (x right, y down, z forward).

    No y flip: image row v maps to framebuffer row v, so the framebuffer read back from
    the bottom row up is already in image order (top row first)."""
    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5  # GL pixel centres at +0.5
    return np.array([
        [2 * fx / w, 0, 2 * cx / w - 1, 0],
        [0, 2 * fy / h, 2 * cy / h - 1, 0],
        [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, 1, 0],
    ])


def shadow_matrix(light_dir: np.ndarray, z: float = 0.001) -> np.ndarray:
    """Projects points along the light direction onto the plane z = `z` (world)."""
    lx, ly, lz = light_dir / np.linalg.norm(light_dir)
    return np.array([[1, 0, -lx / lz, lx / lz * z], [0, 1, -ly / lz, ly / lz * z], [0, 0, 0, z], [0, 0, 0, 1]])


class GLRenderer:
    def __init__(self, cam: Camera, render_scale: float = 1.0, samples: int = 4, near: float = 0.05,
                 far: float = 100.0, light_dir=(0.3, 0.2, 1.0), ambient: float = 0.45,
                 shadow_alpha: float = 0.35, max_half_angle_deg: float = 80.0):
        self.cam = cam
        self.light_dir = np.asarray(light_dir, float)
        self.ambient, self.shadow_alpha = ambient, shadow_alpha
        w, h = cam.image_size

        # Normalised coordinates of every real pixel -> extent of the virtual pinhole image.
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pix = np.column_stack([u.ravel(), v.ravel()])
        norm = undistort_points(cam.model, pix, cam.K, cam.dist)
        # The iterative undistortion can fail at the edge of the calibrated field of view: keep only
        # pixels that project back onto themselves; the rest stay transparent.
        back = project_points(cam.model, np.column_stack([norm, np.ones(len(norm))]), np.zeros(3), np.zeros(3),
                              cam.K, cam.dist)
        limit = np.tan(np.radians(max_half_angle_deg))
        valid = (np.linalg.norm(back - pix, axis=1) < 1.0) & (np.abs(norm) < limit).all(axis=1)
        self.invalid_fraction = float(1 - valid.mean())
        lo, hi = norm[valid].min(axis=0), norm[valid].max(axis=0)
        f = float(np.mean([cam.K[0, 0], cam.K[1, 1]])) * render_scale
        self.virtual_size = (int(np.ceil((hi[0] - lo[0]) * f)) + 2, int(np.ceil((hi[1] - lo[1]) * f)) + 2)
        self.K_virtual = np.array([[f, 0, -lo[0] * f + 1], [0, f, -lo[1] * f + 1], [0, 0, 1]])
        mx = norm[:, 0] * f + self.K_virtual[0, 2]
        my = norm[:, 1] * f + self.K_virtual[1, 2]
        mx[~valid], my[~valid] = -1, -1  # outside the virtual image -> border value (transparent)
        self.map_x = mx.reshape(h, w).astype(np.float32)
        self.map_y = my.reshape(h, w).astype(np.float32)

        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vs = self.virtual_size
        samples = min(samples, self.ctx.max_samples)
        self.fbo_ms = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(vs, 4, samples=samples)],
            depth_attachment=self.ctx.depth_renderbuffer(vs, samples=samples))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(vs, 4)])
        self.P = projection_from_K(self.K_virtual, vs, near, far)
        self._vaos: dict[int, moderngl.VertexArray] = {}

    def _vao(self, mesh: Mesh) -> moderngl.VertexArray:
        key = id(mesh)
        if key not in self._vaos:
            data = np.hstack([mesh.vertices, mesh.normals]).astype("f4")
            vbo = self.ctx.buffer(data.tobytes())
            self._vaos[key] = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_pos", "in_normal")])
        return self._vaos[key]

    def _draw(self, mesh: Mesh, T: np.ndarray, color, shade: bool) -> None:
        mvp = self.P @ self.cam.T_cam_world @ T
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())  # GL is column-major
        self.prog["model"].write(T.T.astype("f4").tobytes())
        self.prog["color"].value = tuple(float(c) for c in color)
        self.prog["shade"].value = shade
        self._vao(mesh).render(moderngl.TRIANGLES)

    def render(self, items: list[DrawItem], holdouts: list[DrawItem] = ()) -> np.ndarray:
        """Premultiplied RGBA uint8 image (H, W, 4) in the real (distorted) camera image."""
        ctx = self.ctx
        self.fbo_ms.use()
        ctx.clear(0, 0, 0, 0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA  # premultiplied "over"
        self.prog["light_dir"].value = tuple(self.light_dir)
        self.prog["cam_pos"].value = tuple(self.cam.T_world_cam[:3, 3])
        self.prog["ambient"].value = self.ambient

        # 1) holdouts: depth only
        ctx.disable(moderngl.BLEND)
        self.fbo_ms.color_mask = (False, False, False, False)
        for it in holdouts:
            self._draw(it.mesh, it.T_world_obj, (0, 0, 0, 0), False)
        self.fbo_ms.color_mask = (True, True, True, True)

        # 2) opaque objects, 3) shadows (no depth write), 4) transparent objects back to front
        opaque = [it for it in items if it.color[3] >= 0.999]
        transparent = [it for it in items if it.color[3] < 0.999]
        for it in opaque:
            self._draw(it.mesh, it.T_world_obj, it.color, it.shade)
        ctx.enable(moderngl.BLEND)
        ctx.depth_mask = False
        S = shadow_matrix(self.light_dir)
        for it in items:
            if it.shadow:
                self._draw(it.mesh, S @ it.T_world_obj, (0, 0, 0, self.shadow_alpha), False)
        cam_pos = self.cam.T_world_cam[:3, 3]
        transparent.sort(key=lambda it: -np.linalg.norm(it.T_world_obj[:3, 3] - cam_pos))
        for it in transparent:
            self._draw(it.mesh, it.T_world_obj, it.color, it.shade)
        ctx.depth_mask = True

        ctx.copy_framebuffer(self.fbo, self.fbo_ms)  # resolve multisampling
        vw, vh = self.virtual_size
        img = np.frombuffer(self.fbo.read(components=4, dtype="f1"), np.uint8).reshape(vh, vw, 4)
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        self.ctx.release()


def composite(frame_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA overlay over a BGR frame."""
    rgb = overlay_rgba[..., :3].astype(np.float32)
    a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb[..., ::-1] + frame_bgr.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)
