"""Background Vicon stream with an explicit state machine.

    idle -> connecting -> streaming -> (error -> connecting ...) -> stopped

Every frame is stored with a host timestamp (receive time minus the SDK's reported
latency, i.e. an estimate of the Vicon exposure time on the host clock).
"""

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from calib.transforms import interpolate_pose
from record.subscription import Subscription
from record.vicon_client import ViconClient, ViconError


class State(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ViconSample:
    t_host: float  # time.time() of the exposure (receive time - latency)
    frame_number: int
    t_recv: float = 0.0  # time.time() when the SDK delivered the frame
    latency_s: float = 0.0  # SDK-reported latency (exposure -> delivery)
    poses: dict[str, np.ndarray | None] = field(default_factory=dict)  # subject -> T_world_subject, None if occluded
    markers: np.ndarray | None = None  # (N, 3) all visible markers (labelled + unlabelled) [m], if enabled


class ViconStream:
    def __init__(self, host: str, sdk_dir: str, subjects: list[str], stream_mode: str = "ServerPush",
                 buffer_s: float = 10.0, retry_s: float = 2.0, with_markers: bool = False):
        self.host, self.sdk_dir, self.subjects = host, sdk_dir, list(subjects)
        self.with_markers = with_markers
        self.stream_mode, self.retry_s = stream_mode, retry_s
        self.buffer: deque[ViconSample] = deque(maxlen=int(buffer_s * 400))  # up to 400 Hz
        self.state = State.IDLE
        self.last_error = ""
        self.frame_rate = 0.0
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- public -----------------------------------------------------------------
    def start(self) -> "ViconStream":
        self._thread = threading.Thread(target=self._run, name="vicon", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def latest(self) -> ViconSample | None:
        with self._lock:
            return self.buffer[-1] if self.buffer else None

    def window(self, t_from: float, t_to: float) -> list[ViconSample]:
        with self._lock:
            return [s for s in self.buffer if t_from <= s.t_host <= t_to]

    def nearest(self, t: float) -> ViconSample | None:
        """Buffered sample closest to host time t (e.g. a delayed camera frame time)."""
        with self._lock:
            buf = list(self.buffer)
        if not buf:
            return None
        i = bisect.bisect_left([s.t_host for s in buf], t)
        cands = buf[max(0, i - 1): i + 1]
        return min(cands, key=lambda s: abs(s.t_host - t))

    def pose_at(self, subject: str, t: float, max_gap_s: float = 0.05) -> np.ndarray | None:
        """T_world_subject interpolated at host time t from the buffer; None if not available
        (outside the buffer, occluded around t, or a gap longer than max_gap_s)."""
        with self._lock:
            buf = list(self.buffer)
        times = [s.t_host for s in buf]
        i = bisect.bisect_left(times, t)
        if i == len(buf) and buf and t - times[-1] <= max_gap_s:
            return buf[-1].poses.get(subject)  # slightly newer than the newest sample: hold it
        if i == 0 or i == len(buf):
            return None
        s0, s1 = buf[i - 1], buf[i]
        T0, T1 = s0.poses.get(subject), s1.poses.get(subject)
        if T0 is None or T1 is None or s1.t_host - s0.t_host > max_gap_s:
            return None
        return interpolate_pose(T0, T1, (t - s0.t_host) / (s1.t_host - s0.t_host))

    def subscribe(self, maxsize: int = 5000) -> Subscription:
        """Queue with every sample, e.g. for recording."""
        sub = Subscription(maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.remove(sub)

    def wait_streaming(self, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.state == State.STREAMING and self.latest() is not None:
                return True
            time.sleep(0.05)
        return False

    # --- state machine ----------------------------------------------------------
    def _set(self, state: State, error: str = "") -> None:
        if state != self.state or error:
            print(f"[vicon] {self.state.value} -> {state.value}" + (f": {error}" if error else ""))
        self.state, self.last_error = state, error or self.last_error

    def _run(self) -> None:
        client = ViconClient(self.sdk_dir)
        try:
            while not self._stop.is_set():
                if self.state in (State.IDLE, State.ERROR):
                    self._set(State.CONNECTING)
                if self.state == State.CONNECTING:
                    try:
                        client.connect(self.host, self.stream_mode)
                        client.get_frame()
                        self.frame_rate = client.frame_rate()
                        missing = set(self.subjects) - set(client.subjects())
                        if missing:
                            print(f"[vicon] WARNING subjects not in stream: {sorted(missing)}")
                        self._set(State.STREAMING)
                    except ViconError as e:
                        client.disconnect()
                        self._set(State.ERROR, str(e))
                        self._stop.wait(self.retry_s)
                        continue
                try:
                    self._read_frame(client)
                except ViconError as e:
                    client.disconnect()
                    self._set(State.ERROR, str(e))
                    self._stop.wait(self.retry_s)
        finally:
            client.close()
            self._set(State.STOPPED)

    def _read_frame(self, client: ViconClient) -> None:
        client.get_frame()
        t_recv = time.time()
        latency = client.latency_s()
        sample = ViconSample(t_host=t_recv - latency, frame_number=client.frame_number(), t_recv=t_recv,
                             latency_s=latency)
        for subject in self.subjects:
            try:
                pose = client.segment_pose(subject)
                sample.poses[subject] = None if pose.occluded else pose.T_world_segment
            except ViconError:  # subject not in this frame (disabled in Tracker)
                sample.poses[subject] = None
        if self.with_markers:
            pts = [client.unlabeled_markers()]
            for subject in self.subjects:
                try:
                    pts.append(np.array([m for m in client.markers(subject).values() if m is not None]).reshape(-1, 3))
                except ViconError:
                    pass
            sample.markers = np.vstack(pts)
        with self._lock:
            self.buffer.append(sample)
            subs = list(self._subs)
        for sub in subs:
            sub.put(sample)
