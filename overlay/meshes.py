"""Simple triangle meshes for overlay objects (flat shaded: every face has its own vertices)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float32, object frame [m]
    normals: np.ndarray  # (N, 3) float32

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
