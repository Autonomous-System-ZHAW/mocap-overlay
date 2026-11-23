"""Background camera grabber with an explicit state machine.

    idle -> connecting -> streaming -> (error -> connecting ...) -> stopped

The thread reads frames as fast as the source delivers them, so the decoder never
builds up a backlog (which is what makes a slow `cap.read()` loop lag seconds behind).
Every frame gets the host receive time and the stream's own timestamp (pts).

Consumers:
  read()       newest frame not yet returned (skips older ones), for previews / live use
  subscribe()  queue with every frame, for recording

Video files are played back in real time (paced by their fps), so recordings can
stand in for the live camera during development.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from record.stream import is_file, redact, resolve_source, try_open
from record.subscription import Subscription


class State(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class CameraFrame:
    index: int  # running frame number since start() (continues across reconnects)
    t_host: float  # time.time() when the decoded frame was received
    pts_ms: float  # stream timestamp from the decoder [ms], NaN if unknown
    image: np.ndarray


class CameraStream:
    def __init__(self, camera_cfg: dict, source: str | None = None, retry_s: float = 5.0,
                 max_retry_s: float | None = None):
        """max_retry_s: give up (state STOPPED) after failing this long; None = retry forever."""
        self.source = resolve_source(camera_cfg, source)
        self.retry_s, self.max_retry_s = retry_s, max_retry_s
        self.state = State.IDLE
        self.last_error = ""
        self.fps = 0.0  # measured, exponential moving average
        self.size: tuple[int, int] | None = None
        self._latest: CameraFrame | None = None
        self._last_read_index = -1
        self._subs: list[Subscription] = []
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- public -----------------------------------------------------------------
    def start(self) -> "CameraStream":
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        cv2.destroyAllWindows()
        if exc_type is KeyboardInterrupt:
            print("interrupted")
            return True  # clean exit on Ctrl+C; the capture is released, so the camera gets a TEARDOWN
        return False

    def read(self, timeout: float = 10.0) -> CameraFrame | None:
        """Newest frame that read() has not returned yet. None if the stream stopped or timed out."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._latest is None or self._latest.index == self._last_read_index:
                if self.state == State.STOPPED:
                    return None
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(min(remaining, 0.1))
            self._last_read_index = self._latest.index
            return self._latest

    def subscribe(self, maxsize: int = 250) -> Subscription:
        sub = Subscription(maxsize)
        with self._cond:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._cond:
            self._subs.remove(sub)

    # --- state machine ----------------------------------------------------------
    def _set(self, state: State, error: str = "") -> None:
        if state != self.state:
            print(f"[camera] {self.state.value} -> {state.value}" + (f": {error}" if error else ""))
        with self._cond:
            self.state, self.last_error = state, error or self.last_error
            self._cond.notify_all()

    def _run(self) -> None:
        index, cap, failing_since = 0, None, None
        fps_file, t_start = 0.0, time.monotonic()
        try:
            while not self._stop.is_set():
                if self.state in (State.IDLE, State.ERROR):
                    self._set(State.CONNECTING)
                if self.state == State.CONNECTING:
                    cap = try_open(self.source)
                    if cap is None and is_file(self.source):
                        self._set(State.STOPPED, f"cannot open file {self.source}")  # retrying will not help
                        return
                    if cap is None:
                        failing_since = failing_since or time.monotonic()
                        if self.max_retry_s is not None and time.monotonic() - failing_since > self.max_retry_s:
                            self._set(State.STOPPED, f"cannot open {redact(self.source)}")
                            return
                        self._set(State.ERROR, f"cannot open {redact(self.source)}, retrying in {self.retry_s:.0f} s")
                        self._stop.wait(self.retry_s)
                        continue
                    failing_since = None
                    fps_file = cap.get(cv2.CAP_PROP_FPS) if is_file(self.source) else 0.0
                    t_start = time.monotonic()
                    self._set(State.STREAMING)

                ok, image = cap.read()
                t_host = time.time()
                if not ok:
                    cap.release()
                    cap = None
                    if is_file(self.source):
                        self._set(State.STOPPED, "end of file")
                        return
                    self._set(State.ERROR, "stream dropped")
                    continue

                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame = CameraFrame(index, t_host, pts if pts > 0 or index == 0 else float("nan"), image)
                self._publish(frame)
                index += 1
                if fps_file > 0:  # pace file playback like a live camera
                    self._stop.wait(max(0.0, t_start + index / fps_file - time.monotonic()))
        finally:
            if cap is not None:
                cap.release()
            if self.state != State.STOPPED:
                self._set(State.STOPPED)

    def _publish(self, frame: CameraFrame) -> None:
        with self._cond:
            if self._latest is not None:
                dt = frame.t_host - self._latest.t_host
                if dt > 0:
                    self.fps = 1 / dt if self.fps == 0 else 0.95 * self.fps + 0.05 / dt
            self.size = (frame.image.shape[1], frame.image.shape[0])
            self._latest = frame
            subs = list(self._subs)
            self._cond.notify_all()
        for sub in subs:
            sub.put(frame)
