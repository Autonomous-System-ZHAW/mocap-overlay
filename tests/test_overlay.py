"""Overlay renderer: GL rendering must land exactly where Camera.project() puts points."""

import numpy as np
import pytest

from calib.camera import Camera
from calib.transforms import inv_T, make_T
from overlay.meshes import box, from_spec
from overlay.scene import Scene

SIZE = (1280, 720)
K = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])


def look_at(cam_pos, target):
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    return inv_T(make_T(np.column_stack([x, np.cross(z, x), z]), cam_pos))


@pytest.fixture(scope="module")
def gl():
    try:
        import moderngl
        moderngl.create_standalone_context(backend="egl").release()
    except Exception as e:  # no GPU / EGL on this machine
        pytest.skip(f"no EGL context: {e}")
    from overlay import gl_renderer
    return gl_renderer


@pytest.mark.parametrize("model,dist", [("pinhole", np.zeros(5)), ("pinhole", np.array([-0.3, 0.1, 0, 0, -0.02])),
                                        ("fisheye", np.array([-0.02, 0.03, -0.03, 0.01]))])
def test_render_matches_projection(gl, model, dist):
    cam = Camera(model, SIZE, K, dist, look_at(np.array([0.0, -3.0, 2.0]), np.array([0.0, 0.0, 0.0])))
    r = gl.GLRenderer(cam, samples=1)
    try:
        # Small cubes at several places, also near the image borders where distortion is strong.
        for target in ([0, 0, 0], [1.6, 0.8, 0], [-1.8, 1.5, 0], [1.2, -1.2, 0.3]):
            T = make_T(np.eye(3), target)
            img = r.render([gl.DrawItem(box([0.06, 0.06, 0.06]), T, (1, 1, 1, 1), shade=False)])
            alpha = img[..., 3].astype(float)
            assert alpha.sum() > 0, f"cube at {target} not rendered"
            v, u = np.indices(alpha.shape)
            centroid = np.array([(u * alpha).sum(), (v * alpha).sum()]) / alpha.sum()
            expected, ok = cam.project(np.array([[target[0], target[1], target[2] + 0.03]]))
            assert ok[0]
            assert np.linalg.norm(centroid - expected[0]) < 1.5, (target, centroid, expected[0])
    finally:
        r.release()


def test_holdout_hides_objects_behind(gl):
    cam = Camera("pinhole", SIZE, K, np.zeros(5), look_at(np.array([0.0, -3.0, 0.1]), np.array([0.0, 0.0, 0.1])))
    r = gl.GLRenderer(cam, samples=1)
    try:
        far = gl.DrawItem(box([0.2, 0.2, 0.2]), make_T(np.eye(3), [0, 1, 0]), (1, 0, 0, 1))
        near_holdout = gl.DrawItem(box([0.5, 0.5, 0.5]), make_T(np.eye(3), [0, 0, 0]), (0, 0, 0, 0))
        assert r.render([far])[..., 3].max() == 255
        assert r.render([far], [near_holdout])[..., 3].max() == 0
    finally:
        r.release()


def test_composite_premultiplied(gl):
    frame = np.full((2, 2, 3), 100, np.uint8)
    over = np.zeros((2, 2, 4), np.uint8)
    over[0, 0] = [255, 0, 0, 255]  # opaque red (RGB)
    over[0, 1] = [64, 0, 0, 128]  # half transparent, premultiplied
    out = gl.composite(frame, over)
    assert out[0, 0].tolist() == [0, 0, 255]  # BGR
    assert out[1, 1].tolist() == [100, 100, 100]
    assert abs(int(out[0, 1, 2]) - (64 + 50)) <= 1


def test_scene_yaml(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("""
name: t
origin: {xyz: [1, 2, 0]}
objects:
  - {name: c, mesh: {type: traffic_cone}, pose: {xyz: [0.5, 0, 0]}, repeat: {count: 3, step: [0.1, 0, 0]}}
  - {name: a, attach: car, mesh: {type: box, size: [0.1, 0.1, 0.1]}, pose: {xyz: [0.2, 0, 0]}}
trails: [{subject: car, seconds: 2}]
""")
    s = Scene.from_yaml(p)
    assert [o.name for o in s.objects] == ["c_0", "c_1", "c_2", "a"] and s.subjects == {"car"}
    np.testing.assert_allclose(s.objects[2].T[:3, 3], [1.7, 2, 0])  # origin applied to static objects
    items, _ = s.draw_items({"car": make_T(np.eye(3), [5, 0, 0])})
    np.testing.assert_allclose(items[-1].T_world_obj[:3, 3], [5.2, 0, 0])  # attached: subject pose @ offset
    items, _ = s.draw_items({"car": None})
    assert len(items) == 3  # untracked subject: attached object hidden
    assert from_spec({"type": "cylinder", "radius": 0.1, "height": 0.2}).vertices.shape[1] == 3


def test_vicon_stream_pose_at_interpolates_buffer():
    from record.vicon_stream import ViconSample, ViconStream

    vs = ViconStream("x:801", "unused", ["car"])
    for k in range(10):
        T = None if k == 6 else make_T(np.eye(3), [k * 0.01, 0, 0])
        vs.buffer.append(ViconSample(100 + k * 0.005, k, poses={"car": T}))
    np.testing.assert_allclose(vs.pose_at("car", 100.0125)[:3, 3], [0.025, 0, 0])
    assert vs.pose_at("car", 100.0275) is None  # neighbour occluded
    assert vs.pose_at("car", 99.0) is None  # before the buffer
    np.testing.assert_allclose(vs.pose_at("car", 100.05)[:3, 3], [0.09, 0, 0])  # just after the newest: hold
    assert vs.nearest(100.0124).frame_number == 2
