"""Build the mocap-overlay Blender scene. Runs INSIDE Blender (4.2+), started by
scripts/blender_setup.py:

    blender -b --python blender/setup_scene.py -- <setup.json> <out.blend>

Creates:
  - collection "mocap (generated)": calibrated camera with the undistorted camera video as
    background, one Empty per Vicon subject animated with the recorded poses (delay applied,
    Blender frame 1 = video frame 0), Vicon origin, sun, floor shadow catcher
  - collection "models": empty, for your own objects. Static objects: model in the Vicon
    world. Moving objects: parent them to a subject Empty (it is scaled to 0 while the
    subject is not tracked, which hides its children). Sub-collection "holdouts" (holdout on):
    stand-ins for real objects that hide virtual objects behind them
  - scene "output" (1920x1080, the real camera image): its compositor takes the render of
    scene "mocap", applies the real lens distortion (Map UV) and puts it over the original
    camera video. Render this scene for the final images / video.
Self-contained on purpose: Blender has its own Python, the project's modules are not available.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

GEN = "mocap (generated)"
MODELS = "models"


def setup_engine(scene, cfg):
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    w, h = cfg["virtual"]["size"]
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.fps, scene.render.fps_base = round(cfg["fps"]), round(cfg["fps"]) / cfg["fps"]
    scene.frame_start, scene.frame_end = 1, cfg["frame_count"]
    scene.render.film_transparent = True
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.filepath = "//render/frame_"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"  # keep video colours unchanged (no Filmic/AgX)


def add_camera(scene, cfg, col, base: Path):
    K = np.array(cfg["virtual"]["K"])
    w, h = cfg["virtual"]["size"]
    data = bpy.data.cameras.new("axis_camera")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = K[0, 0] * data.sensor_width / w
    # Principal point (OpenCV pixel centres at integers -> +0.5), shift in units of the image width.
    data.shift_x = (w / 2 - (K[0, 2] + 0.5)) / w
    data.shift_y = ((K[1, 2] + 0.5) - h / 2) / w
    data.clip_start, data.clip_end = 0.05, 200.0
    cam = bpy.data.objects.new("axis_camera", data)
    col.objects.link(cam)
    # OpenCV camera (x right, y down, z forward) -> Blender camera (x right, y up, looks along -z)
    T = np.array(cfg["T_world_cam"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    cam.matrix_world = Matrix(T.tolist())
    cam.lock_location = cam.lock_rotation = (True, True, True)
    cam.lock_scale = (True, True, True)
    scene.camera = cam

    clip = bpy.data.movieclips.load(str(base / cfg["files"]["background"]))
    clip.name = "camera_undistorted"
    clip.frame_start = 1
    bg = data.background_images.new()
    bg.source, bg.clip = "MOVIE_CLIP", clip
    bg.use_camera_clip = False
    bg.alpha, bg.display_depth, bg.frame_method = 1.0, "BACK", "STRETCH"
    data.show_background_images = True
    return cam


def _fill(fc, frames, values, interpolation="LINEAR"):
    """Replace all keyframes of an F-curve in one go."""
    fc.keyframe_points.clear()
    fc.keyframe_points.add(len(frames))
    fc.keyframe_points.foreach_set("co", np.column_stack([frames, values]).ravel().astype(np.float32))
    for kp in fc.keyframe_points:
        kp.interpolation = interpolation
    fc.update()


def add_subject(name: str, T: np.ndarray, col):
    """Empty following a Vicon subject. T: (N, 4, 4), NaN where not tracked."""
    e = bpy.data.objects.new(name, None)
    e.empty_display_type, e.empty_display_size = "ARROWS", 0.2
    e.rotation_mode = "QUATERNION"
    col.objects.link(e)
    valid = np.isfinite(T).all(axis=(1, 2))
    if not valid.any():
        e.scale = (0, 0, 0)
        return e
    # Hold the last valid pose through gaps (scale 0 hides the children there).
    idx = np.maximum.accumulate(np.where(valid, np.arange(len(T)), -1))
    idx[idx < 0] = np.flatnonzero(valid)[0]
    T = T[idx]
    loc = T[:, :3, 3]
    quat = np.array([Matrix(Ti[:3, :3].tolist()).to_quaternion()[:] for Ti in T])  # w, x, y, z
    for i in range(1, len(quat)):  # continuous sign for clean interpolation
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    frames = np.arange(1, len(T) + 1, dtype=np.float32)
    # keyframe_insert creates the action with a correctly bound slot (layered actions, Blender 4.4+);
    # the F-curves are then filled with all frames at once.
    for path in ("location", "rotation_quaternion", "scale"):
        e.keyframe_insert(path, frame=1)
    act = e.animation_data.action
    act.name = f"{name}_vicon"
    channels = {"location": loc, "rotation_quaternion": quat, "scale": np.repeat(valid[:, None], 3, axis=1)}
    for fc in act.fcurves:
        values = channels[fc.data_path][:, fc.array_index].astype(np.float32)
        _fill(fc, frames, values, "CONSTANT" if fc.data_path == "scale" else "LINEAR")
    e["vicon_subject"] = name
    return e


def add_lights_and_floor(scene, cfg, col):
    light_dir = Vector(cfg["light_dir"]).normalized()
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = light_dir.to_track_quat("Z", "Y")  # the sun shines along its -Z
    col.objects.link(sun)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.05, 1)
    scene.world = world

    s = 6.0  # 12 x 12 m floor, z = 0
    mesh = bpy.data.meshes.new("floor")
    mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("floor_shadow_catcher", mesh)
    col.objects.link(floor)
    floor.is_shadow_catcher = True  # Cycles
    floor.hide_select = True
    floor.data.materials.append(shadow_catcher_material(lit=3.0 * light_dir.z / math.pi + 0.05, ambient=0.05))

    origin = bpy.data.objects.new("vicon_origin", None)
    origin.empty_display_type, origin.empty_display_size = "ARROWS", 0.5
    col.objects.link(origin)


def shadow_catcher_material(lit: float, ambient: float, strength: float = 0.5):
    """EEVEE shadow catcher: transparent where fully lit, dark where shadowed.
    Tweak the 'shadow range' Map Range node (lit / shadowed brightness) and 'shadow strength'."""
    m = bpy.data.materials.new("shadow_catcher")
    m.use_nodes = True
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    else:
        m.blend_method = "BLEND"
    nt = m.node_tree
    nt.nodes.clear()
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    to_rgb = nt.nodes.new("ShaderNodeShaderToRGB")
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.label = "shadow range"
    rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = ambient, lit
    rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 1.0, 0.0
    mul = nt.nodes.new("ShaderNodeMath")
    mul.label = "shadow strength"
    mul.operation, mul.inputs[1].default_value = "MULTIPLY", strength
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    black = nt.nodes.new("ShaderNodeEmission")
    black.inputs["Color"].default_value = (0, 0, 0, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    links = nt.links
    links.new(diffuse.outputs[0], to_rgb.inputs[0])
    links.new(to_rgb.outputs["Color"], bw.inputs[0])
    links.new(bw.outputs[0], rng.inputs["Value"])
    links.new(rng.outputs[0], mul.inputs[0])
    links.new(mul.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(black.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs["Surface"])
    for i, n in enumerate([diffuse, to_rgb, bw, rng, mul, mix, out]):
        n.location = (i * 220, 0)
    transparent.location, black.location = (5 * 220, 200), (5 * 220, -200)
    return m


def uv_map_image(base: Path, cfg) -> bpy.types.Image:
    """Lens map as a float EXR image: for each real pixel the UV to sample in the render."""
    uv = np.load(base / cfg["files"]["distort_uv"])  # (H, W, 2), row 0 = top of the image
    h, w = uv.shape[:2]
    rgba = np.zeros((h, w, 4), np.float32)
    # Map UV reads UV from R/G and uses B as coverage (like Blender's UV pass): 1 = valid pixel.
    rgba[..., :2], rgba[..., 2], rgba[..., 3] = uv, (uv[..., 0] >= 0).astype(np.float32), 1.0
    img = bpy.data.images.new("lens_distortion_uv", w, h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.flipud(rgba).ravel())  # Blender stores rows bottom-up
    img.filepath_raw = str(base / "lens_distortion_uv.exr")
    img.file_format = "OPEN_EXR"
    img.save()
    return img


def setup_output_scene(mocap, cfg, base: Path):
    """Scene 'output' at the real camera resolution. Its compositor renders scene 'mocap'
    (CG at the virtual pinhole resolution), applies the real lens distortion and puts the
    result over the original camera video."""
    out = bpy.data.scenes.new("output")
    w, h = cfg["image_size"]
    out.render.resolution_x, out.render.resolution_y, out.render.resolution_percentage = w, h, 100
    out.render.fps, out.render.fps_base = mocap.render.fps, mocap.render.fps_base
    out.frame_start, out.frame_end = mocap.frame_start, mocap.frame_end
    out.render.engine = mocap.render.engine
    out.render.film_transparent = True
    out.view_settings.view_transform = "Standard"
    out.render.filepath = "//render/output_"
    out.render.image_settings.file_format = "PNG"
    out.render.image_settings.color_mode = "RGBA"
    out.use_nodes = True
    nt = out.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    rl.scene, rl.label = mocap, "render of scene 'mocap'"
    uv_img = nt.nodes.new("CompositorNodeImage")
    uv_img.image, uv_img.label = uv_map_image(base, cfg), "lens map (do not edit)"
    map_uv = nt.nodes.new("CompositorNodeMapUV")
    map_uv.label = "real lens distortion"
    clip_node = nt.nodes.new("CompositorNodeMovieClip")
    video = bpy.data.movieclips.load(cfg["files"]["video"])
    video.name, video.frame_start = "camera_original", 1
    clip_node.clip, clip_node.label = video, "original camera video"
    over = nt.nodes.new("CompositorNodeAlphaOver")
    switch = nt.nodes.new("CompositorNodeSwitch")
    switch.label = "camera video: off / on"
    if "Switch" in switch.inputs:  # Blender 4.5+: the switch is an input socket
        switch.inputs["Switch"].default_value = True
    else:
        switch.check = True
    comp = nt.nodes.new("CompositorNodeComposite")
    viewer = nt.nodes.new("CompositorNodeViewer")
    links = nt.links
    links.new(rl.outputs["Image"], map_uv.inputs["Image"])
    links.new(uv_img.outputs["Image"], map_uv.inputs["UV"])
    links.new(clip_node.outputs["Image"], over.inputs[1])  # background
    links.new(map_uv.outputs["Image"], over.inputs[2])  # foreground
    links.new(map_uv.outputs["Image"], switch.inputs["Off"])  # CG only, transparent
    links.new(over.outputs["Image"], switch.inputs["On"])  # CG over the camera video
    links.new(switch.outputs["Image"], comp.inputs["Image"])
    links.new(switch.outputs["Image"], viewer.inputs["Image"])
    for node, loc in ((rl, (0, 300)), (uv_img, (0, -50)), (clip_node, (0, -350)), (map_uv, (300, 200)),
                      (over, (550, -100)), (switch, (800, 100)), (comp, (1050, 200)), (viewer, (1050, -50))):
        node.location = loc
    return out


CALIBRATION_TOOL = '''"""Calibrate the EEVEE shadow catcher of the floor.

Run again (Text editor > Run Script) after changing the sun or the world light. Renders the
floor once lit and once without the sun, and sets the "shadow range" node of the material:
fully lit floor -> transparent, floor in shadow -> dark. Cycles uses the real shadow catcher
(Object > Visibility > Shadow Catcher) and needs no calibration.
"""
import os
import tempfile

import bpy
import numpy as np


def _floor_brightness(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=scene.name)
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)  # rows bottom-up
    bpy.data.images.remove(img)
    region = px[int(h * 0.1):int(h * 0.4), int(w * 0.4):int(w * 0.6), :3]  # floor, lower middle
    return float(np.median(region @ np.array([0.2126, 0.7152, 0.0722], np.float32)))


def calibrate(scene=None):
    scene = scene or bpy.data.scenes["mocap"]
    floor, sun = bpy.data.objects["floor_shadow_catcher"], bpy.data.objects["sun"]
    nt = floor.active_material.node_tree
    rng = next(n for n in nt.nodes if n.label == "shadow range")
    bw = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeRGBToBW")
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    surface = out.inputs["Surface"].links[0].from_socket
    emit = nt.nodes.new("ShaderNodeEmission")  # temporarily show the floor brightness
    nt.links.new(bw.outputs[0], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])

    r, im = scene.render, scene.render.image_settings
    saved = (r.resolution_percentage, r.film_transparent, r.filepath, r.use_compositing,
             im.file_format, im.color_mode, im.color_depth)
    hidden = {o.name: o.hide_render for o in scene.objects}
    tmp = tempfile.mkdtemp()
    try:
        for o in scene.objects:
            o.hide_render = o not in (floor, sun)
        r.resolution_percentage, r.film_transparent, r.use_compositing = 10, False, False
        im.file_format, im.color_mode, im.color_depth = "OPEN_EXR", "RGBA", "32"
        lit = _floor_brightness(scene, os.path.join(tmp, "lit.exr"))
        sun.hide_render = True
        ambient = _floor_brightness(scene, os.path.join(tmp, "ambient.exr"))
    finally:
        (r.resolution_percentage, r.film_transparent, r.filepath, r.use_compositing,
         im.file_format, im.color_mode, im.color_depth) = saved
        for o in scene.objects:
            o.hide_render = hidden.get(o.name, o.hide_render)
        nt.links.new(surface, out.inputs["Surface"])
        nt.nodes.remove(emit)
    if lit - ambient < 1e-3:
        raise RuntimeError(f"floor not lit by the sun (lit {lit:.4f}, ambient {ambient:.4f})")
    rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = ambient, lit
    print(f"shadow catcher calibrated: lit floor {lit:.4f}, shadowed floor {ambient:.4f}")


if __name__ == "__main__":
    calibrate()
'''


def calibrate_shadow_catcher(scene):
    ns = {"__name__": "mocap_calibration"}
    exec(CALIBRATION_TOOL, ns)
    ns["calibrate"](scene)


def readme_text(cfg) -> str:
    return f"""mocap-overlay scene (generated by scripts/blender_setup.py)

recording: {cfg['recording']}
calibration: {cfg['calibration']}
camera delay applied: {cfg['delay_s'] * 1000:.0f} ms
frames: 1..{cfg['frame_count']} = video frames 0..{cfg['frame_count'] - 1} at {cfg['fps']:.2f} fps

Collections
  mocap (generated)  camera, Vicon subject Empties, sun, floor shadow catcher, Vicon origin.
                     Do not move the camera or the Empties (their animation is the Vicon data).
  models             your objects. Static: model in place (Vicon world, metres, z up).
                     Moving with a Vicon subject: parent to the subject Empty
                     (subject frame, e.g. car: origin rear axle centre, x forward).
    holdouts         stand-ins for real objects (e.g. a box the size of the real car, parented
                     to its Empty): invisible, but they hide virtual objects behind them.

Export for the live view / offline renderer (save first):
  uv run python scripts/blender_export.py data/blender/<recording> --name <scene name>

Scenes
  mocap   where you work. Camera view (Numpad 0) shows the camera video undistorted: the Blender
          camera is an ideal pinhole camera with a larger image than the real one. Rendering this
          scene gives the CG alone in that pinhole image (//render/frame_####.png).
  output  1920x1080 like the real camera. Its compositor renders scene "mocap", applies the real
          lens distortion (Map UV with the lens map) and puts the result over the original video.
          Render THIS scene for final images / videos (//render/output_####.png).
          Switch node "camera video: off / on": off = CG only with transparent background.
          Command line: blender -b mocap_scene.blend -S output -a

Shadows on the real floor
  EEVEE: the floor uses a shadow catcher material. After changing the sun or the world light,
         run the text block "calibrate_shadow_catcher.py" (Text editor > Run Script).
  Cycles: real shadow catcher (already enabled on the floor object).
"""


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    setup_path, out_blend = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    base = setup_path.parent
    cfg = json.loads(setup_path.read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "mocap"
    gen = bpy.data.collections.new(GEN)
    models = bpy.data.collections.new(MODELS)
    scene.collection.children.link(gen)
    scene.collection.children.link(models)
    # Stand-ins for real objects (e.g. the real car): invisible, but hide what is behind them.
    holdouts = bpy.data.collections.new("holdouts")
    models.children.link(holdouts)
    scene.view_layers[0].layer_collection.children[MODELS].children["holdouts"].holdout = True

    setup_engine(scene, cfg)
    add_camera(scene, cfg, gen, base)
    poses = np.load(base / cfg["files"]["poses"])
    for name in cfg["subjects"]:
        add_subject(name, poses[name], gen)
    add_lights_and_floor(scene, cfg, gen)
    calibrate_shadow_catcher(scene)
    setup_output_scene(scene, cfg, base)
    txt = bpy.data.texts.new("README_mocap")
    txt.write(readme_text(cfg))
    tool = bpy.data.texts.new("calibrate_shadow_catcher.py")
    tool.write(CALIBRATION_TOOL)
    scene.frame_set(1)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()
    print(f"saved {out_blend}")


if __name__ == "__main__":
    main()
