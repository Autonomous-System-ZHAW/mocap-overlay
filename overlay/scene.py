"""Overlay scene from YAML: static objects in the Vicon world, objects attached to Vicon
subjects, holdouts (invisible stand-ins for real objects) and trajectory trails.

    origin: {xyz: [-0.35, -1.75, 0.0], rpy_deg: [0, 0, 0]}  # optional, moves all static objects
    objects:
      - name: cone_1
        mesh: {type: traffic_cone, radius: 0.06, height: 0.2}
        pose: {xyz: [1.0, 0.5, 0.0], rpy_deg: [0, 0, 0]}     # in the Vicon world
        color: "#ff7a00"
        opacity: 1.0
        shadow: true
        repeat: {count: 5, step: [0.5, 0, 0]}                 # optional copies along a step
      - name: street
        mesh: {type: gltf, file: assets/demo/world.glb}       # Blender export (scripts/blender_export.py)
      - name: car_body
        attach: athena                                       # follows this Vicon subject
        mesh: {type: box, size: [0.45, 0.22, 0.12]}
        pose: {xyz: [0.18, 0, 0.03]}                           # offset in the subject frame
        color: "#2f7bff"
        opacity: 0.5
    holdouts:
      - attach: athena
        mesh: {type: box, size: [0.5, 0.25, 0.15]}
    trails:
      - subject: athena
        seconds: 3.0
        color: "#00e5ff"
        width: 3
        offset: [0.0, 0.0, 0.0]                               # point in the subject frame
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from calib.transforms import make_T
from overlay.gl_renderer import DrawItem
from overlay.meshes import Part, from_spec, load_gltf


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    return tuple(int(c[i: i + 2], 16) / 255 for i in (0, 2, 4))


def pose_from_spec(spec: dict | None) -> np.ndarray:
    spec = spec or {}
    R = Rotation.from_euler("xyz", spec.get("rpy_deg", [0, 0, 0]), degrees=True).as_matrix()
    return make_T(R, spec.get("xyz", [0, 0, 0]))


@dataclass
class SceneObject:
    name: str
    parts: list[Part]  # meshes with their own material colour / texture (glTF) or none (primitives)
    T: np.ndarray  # world pose (static) or offset in the subject frame (attached)
    color: tuple[float, float, float] | None  # YAML colour: overrides the material colours
    opacity: float = 1.0
    attach: str | None = None
    shadow: bool = False


@dataclass
class Trail:
    subject: str
    seconds: float
    color_bgr: tuple[int, int, int]
    width: int
    offset: np.ndarray


@dataclass
class Scene:
    name: str
    objects: list[SceneObject] = field(default_factory=list)
    holdouts: list[SceneObject] = field(default_factory=list)
    trails: list[Trail] = field(default_factory=list)

    @property
    def subjects(self) -> set[str]:
        return {o.attach for o in self.objects + self.holdouts if o.attach} | {t.subject for t in self.trails}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scene":
        with open(path) as f:
            d = yaml.safe_load(f)
        scene = cls(d.get("name", Path(path).stem))
        origin = pose_from_spec(d.get("origin"))
        for o in d.get("objects", []):
            objs = _expand(o)
            for obj in objs:
                if obj.attach is None:
                    obj.T = origin @ obj.T
            scene.objects += objs
        for o in d.get("holdouts", []):
            scene.holdouts += _expand({**o, "name": o.get("name", f"holdout_{o.get('attach')}")})
        for t in d.get("trails", []):
            r, g, b = hex_to_rgb(t.get("color", "#00e5ff"))
            scene.trails.append(Trail(t["subject"], float(t.get("seconds", 3.0)),
                                      (int(b * 255), int(g * 255), int(r * 255)), int(t.get("width", 3)),
                                      np.asarray(t.get("offset", [0, 0, 0]), float)))
        return scene

    def draw_items(self, poses: dict[str, np.ndarray | None]) -> tuple[list[DrawItem], list[DrawItem]]:
        """Resolve attachments with the current subject poses (None = not tracked: object hidden)."""
        def resolve(objs):
            out = []
            for o in objs:
                if o.attach is None:
                    T = o.T
                elif poses.get(o.attach) is not None:
                    T = poses[o.attach] @ o.T
                else:
                    continue
                for part in o.parts:
                    rgb = o.color or (part.color[:3] if part.color else (1.0, 1.0, 1.0))
                    alpha = (part.color[3] if part.color else 1.0) * o.opacity
                    texture = None if o.color else part.texture
                    out.append(DrawItem(part.mesh, T, (*rgb, alpha), o.shadow, texture=texture))
            return out
        return resolve(self.objects), resolve(self.holdouts)


_GLTF_CACHE: dict[tuple[str, str], list[Part]] = {}


def _parts(spec: dict) -> list[Part]:
    if spec["type"] == "gltf":
        key = (str(Path(spec["file"]).resolve()), spec.get("up", "y"))
        if key not in _GLTF_CACHE:  # load each file once, also when repeated
            _GLTF_CACHE[key] = load_gltf(spec["file"], spec.get("up", "y"))
        return _GLTF_CACHE[key]
    return [Part(from_spec(spec))]


def _expand(o: dict) -> list[SceneObject]:
    parts = _parts(o["mesh"])
    color = hex_to_rgb(o["color"]) if "color" in o else (None if o["mesh"]["type"] == "gltf" else (1.0, 1.0, 1.0))
    base = pose_from_spec(o.get("pose"))
    rep = o.get("repeat", {"count": 1, "step": [0, 0, 0]})
    out = []
    for k in range(int(rep["count"])):
        T = base.copy()
        T[:3, 3] += k * np.asarray(rep["step"], float)
        name = o.get("name", "object") + (f"_{k}" if rep["count"] > 1 else "")
        out.append(SceneObject(name, parts, T, color, float(o.get("opacity", 1.0)), o.get("attach"),
                               bool(o.get("shadow", False))))
    return out
