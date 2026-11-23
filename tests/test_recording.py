"""Camera grabber, recorder and recording loader, using a video file as a stand-in camera."""

import time

import cv2
import numpy as np
import pytest

from record.camera_stream import CameraStream, State
from record.recorder import Recorder, RecState
from record.recording import PoseTrack, Recording
from record.subscription import Subscription
from record.vicon_stream import ViconSample

N_FRAMES, FPS, SIZE = 40, 25, (320, 240)


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    """Frame i is uniformly gray with value 5 * i, so frames can be identified after decoding."""
    path = tmp_path_factory.mktemp("video") / "cam.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    for i in range(N_FRAMES):
        w.write(np.full((SIZE[1], SIZE[0], 3), 5 * i, np.uint8))
    w.release()
    return str(path)


def gray(img) -> int:
    return round(float(img.mean()) / 5)


def test_camera_stream_delivers_every_frame_in_real_time(video_file):
    stream = CameraStream({"rtsp_url": ""}, video_file)
    sub = stream.subscribe()
    t0 = time.monotonic()
    with stream:
        frames = []
        while (f := sub.get(timeout=5)) is not None:
            frames.append(f)
            if len(frames) == N_FRAMES:
                break
    elapsed = time.monotonic() - t0
    assert [f.index for f in frames] == list(range(N_FRAMES))
    assert [gray(f.image) for f in frames] == list(range(N_FRAMES))
    assert elapsed > 0.8 * N_FRAMES / FPS  # paced like a camera, not decoded as fast as possible
    t = np.array([f.t_host for f in frames])
    assert np.all(np.diff(t) > 0)
    assert stream.state == State.STOPPED and sub.dropped == 0


def test_read_returns_newest_frame_and_none_at_end(video_file):
    with CameraStream({"rtsp_url": ""}, video_file) as stream:
        first = stream.read(timeout=5)
        time.sleep(0.5)  # ~12 frames arrive meanwhile
        newer = stream.read(timeout=5)
        assert newer.index > first.index + 2  # skipped the backlog (loose: CI / busy machines)
        while stream.read(timeout=5) is not None:
            pass
        assert stream.state == State.STOPPED


class FakeVicon:
    """Stands in for ViconStream: the test pushes samples into the recorder's subscription."""

    host, subjects, frame_rate = "fake:801", ["car"], 100.0

    def __init__(self):
        self.sub = None

    def subscribe(self, maxsize=5000):
        self.sub = Subscription(maxsize)
        return self.sub

    def unsubscribe(self, sub):
        pass


def yaw(deg: float, pos) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = pos
    return T


def test_recorder_round_trip(video_file, tmp_path):
    camera = CameraStream({"rtsp_url": ""}, video_file)
    vicon = FakeVicon()
    rec = Recorder(tmp_path / "rec", camera, vicon, fps=FPS, meta={"note": "test"})
    with camera:
        assert camera.read(timeout=5) is not None
        rec.start()
        t0 = time.time()
        # Car drives along x at 1 m/s and turns at 90 deg/s, 100 Hz; occluded for samples 20-24.
        for k in range(100):
            t = t0 + k * 0.01
            T = None if 20 <= k < 25 else yaw(90 * k * 0.01, [k * 0.01, 0, 0.05])
            vicon.sub.put(ViconSample(t, 1000 + k, t + 0.003, 0.003, {"car": T}))
        while camera.state != State.STOPPED:
            time.sleep(0.05)
    rec.stop()
    assert rec.state == RecState.DONE, rec.error
    stats = rec.stats()
    assert stats["vicon_samples"] == 100 and stats["frames"] >= N_FRAMES - 3 and stats["frames_dropped"] == 0

    r = Recording(tmp_path / "rec")
    assert r.meta["note"] == "test" and r.meta["stats"]["frames"] == stats["frames"]
    assert len(r.frames) == stats["frames"]
    decoded = list(r.iter_frames())
    assert len(decoded) == len(r.frames)
    first_gray = gray(decoded[0][2])  # recording may start a frame or two into the file
    assert [gray(img) for _, _, img in decoded] == list(range(first_gray, first_gray + len(decoded)))

    T, ok = r.pose_at("car", [t0 + 0.105, t0 + 0.22, t0 + 0.505])
    assert ok.tolist() == [True, False, True]  # middle one falls into the occlusion
    np.testing.assert_allclose(T[0, :3, 3], [0.105, 0, 0.05], atol=1e-6)
    yaw_deg = np.degrees(np.arctan2(T[2, 1, 0], T[2, 0, 0]))
    assert abs(yaw_deg - 90 * 0.505) < 1e-3


def test_pose_track_interpolation_and_gaps():
    t = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    poses = [yaw(0, [0, 0, 0]), yaw(90, [1, 0, 0]), yaw(180, [2, 0, 0]), yaw(270, [3, 0, 0]), yaw(0, [4, 0, 0])]
    from scipy.spatial.transform import Rotation

    track = PoseTrack(t, np.array([T[:3, 3] for T in poses]),
                      np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in poses]), np.zeros(5, bool))
    T, ok = track.at([0.5, 2.25, 5.0, -1.0, 11.0], max_gap_s=2.0)
    assert ok.tolist() == [True, True, False, False, False]  # gap 3..10 too long, outside range
    np.testing.assert_allclose(T[0], yaw(45, [0.5, 0, 0]), atol=1e-9)
    np.testing.assert_allclose(T[1], yaw(202.5, [2.25, 0, 0]), atol=1e-9)  # slerp across the +-180 wrap
