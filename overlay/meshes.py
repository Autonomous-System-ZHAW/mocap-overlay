"""Simple triangle meshes for overlay objects (flat shaded: every face has its own vertices)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float32, object frame [m], 3 consecutive vertices per triangle
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray | None = None  # (N, 2) float32 texture coordinates (v = 0 at the image bottom)

    @staticmethod
    def from_triangles(tris: np.ndarray) -> "Mesh":
        """(M, 3, 3) triangles -> flat-shaded mesh."""
        tris = np.asarray(tris, np.float32)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return Mesh(tris.reshape(-1, 3), np.repeat(n, 3, axis=0).astype(np.float32))


def _quads_to_tris(quads: np.ndarray) -> np.ndarray:
    """(M, 4, 3) counter-clockwise quads -> (2M, 3, 3) triangles."""
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def box(size) -> Mesh:
    """Box with its bottom face centred on the origin (sits on z = 0)."""
    sx, sy, sz = np.asarray(size, float) / [2, 2, 1]
    v = np.array([[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return Mesh.from_triangles(_quads_to_tris(v[np.array(faces)]))


def plane(size) -> Mesh:
    """Rectangle in the z = 0 plane, centred on the origin, facing +z."""
    sx, sy = np.asarray(size, float) / 2
    v = np.array([[[-sx, -sy, 0], [sx, -sy, 0], [sx, sy, 0], [-sx, sy, 0]]])
    return Mesh.from_triangles(_quads_to_tris(v))


def cylinder(radius: float, height: float, segments: int = 32) -> Mesh:
    """Cylinder standing on z = 0."""
    a = np.linspace(0, 2 * np.pi, segments + 1)
    ring = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
    tris = []
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        tris += [[[x0, y0, 0], [x1, y1, 0], [x1, y1, height]], [[x0, y0, 0], [x1, y1, height], [x0, y0, height]],
                 [[0, 0, height], [x0, y0, height], [x1, y1, height]], [[0, 0, 0], [x1, y1, 0], [x0, y0, 0]]]
    return Mesh.from_triangles(np.array(tris))


def cone(radius: float, height: float, segments: int = 32) -> Mesh:
    """Cone standing on z = 0 with its tip at z = height."""
    a = np.linspace(0, 2 * np.pi, segments + 1)
    ring = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
    tris = []
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        tris += [[[x0, y0, 0], [x1, y1, 0], [0, 0, height]], [[0, 0, 0], [x1, y1, 0], [x0, y0, 0]]]
    return Mesh.from_triangles(np.array(tris))


def merge(meshes: list[Mesh]) -> Mesh:
    return Mesh(np.concatenate([m.vertices for m in meshes]), np.concatenate([m.normals for m in meshes]))


def translated(mesh: Mesh, offset) -> Mesh:
    return Mesh(mesh.vertices + np.asarray(offset, np.float32), mesh.normals)


@dataclass
class Part:
    """One drawable piece of a model: mesh + material."""

    mesh: Mesh
    color: tuple[float, float, float, float] | None = None  # display RGBA; None = use the scene colour
    texture: np.ndarray | None = None  # (H, W, 4) uint8 RGBA, row 0 = image top


def _srgb(linear: np.ndarray) -> np.ndarray:
    """Linear (glTF / Blender material values) -> display sRGB, so colours look like in Blender."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


# glTF is Y-up: Blender (x, y, z) is stored as (x, z, -y). Back to Z-up (Vicon / Blender):
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)


def load_gltf(path, up: str = "y") -> list[Part]:
    """Meshes of a .glb/.gltf file with base colour and base colour texture (no other PBR terms)."""
    import trimesh  # only needed for glTF models

    scene = trimesh.load(path, force="scene", process=False)  # keep authored (flat/smooth) normals
    R = Y_UP_TO_Z_UP if up == "y" else np.eye(3, dtype=np.float32)
    parts = []
    for node in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node]
        m = scene.geometry[geom_name]
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        verts = (trimesh.transform_points(m.vertices, T) @ R.T).astype(np.float32)
        normals = (m.vertex_normals @ np.asarray(T)[:3, :3].T @ R.T).astype(np.float32)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        f = m.faces.ravel()
        vis = m.visual
        uv = getattr(vis, "uv", None)
        mat = getattr(vis, "material", None)
        color, texture = (1.0, 1.0, 1.0, 1.0), None
        if mat is not None:
            factor = getattr(mat, "baseColorFactor", None)
            if factor is None and getattr(mat, "diffuse", None) is not None:
                factor = mat.diffuse
            if factor is not None:
                factor = np.asarray(factor, float) / (255.0 if np.asarray(factor).dtype.kind in "ui" else 1.0)
                color = (*_srgb(factor[:3]), float(factor[3]) if len(factor) > 3 else 1.0)
            tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if tex is not None and uv is not None:
                texture = np.asarray(tex.convert("RGBA"), np.uint8)
        elif getattr(vis, "kind", None) == "face" or getattr(vis, "kind", None) == "vertex":
            rgba = np.asarray(vis.main_color, float) / 255.0
            color = (*rgba[:3], rgba[3])
        mesh = Mesh(verts[f], normals[f], None if texture is None else np.asarray(uv, np.float32)[f])
        parts.append(Part(mesh, color, texture))
    if not parts:
        raise ValueError(f"no meshes in {path}")
    return parts


def from_spec(spec: dict) -> Mesh:
    """Mesh from a scene YAML entry, e.g. {type: box, size: [0.5, 0.3, 0.2]}."""
    kind = spec["type"]
    if kind == "box":
        return box(spec["size"])
    if kind == "plane":
        return plane(spec["size"])
    if kind == "cylinder":
        return cylinder(spec["radius"], spec["height"], spec.get("segments", 32))
    if kind == "cone":
        return cone(spec["radius"], spec["height"], spec.get("segments", 32))
    if kind == "traffic_cone":  # cone on a square base plate
        r, h = spec.get("radius", 0.06), spec.get("height", 0.2)
        return merge([box([2.4 * r, 2.4 * r, 0.1 * r]), translated(cone(r, h), [0, 0, 0.1 * r])])
    raise ValueError(f"unknown mesh type {kind!r}")
