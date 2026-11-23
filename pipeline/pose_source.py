"""Where the overlay gets Vicon poses from: the live stream, or a recording (replay).

Both answer the same questions for a given exposure time on the host clock, so the
frame composer does not care whether it runs live or on recorded data.
"""

from typing import Protocol

import numpy as np

from calib.transforms import apply
from record.recording import Recording
from record.vicon_stream import State, ViconStream


class PoseSource(Protocol):
    def pose_at(self, subject: str, t: float) -> np.ndarray | None: ...

    def trail(self, subject: str, t: float, seconds: float, offset: np.ndarray) -> list[np.ndarray]: ...

    def status(self) -> tuple[bool, str]: ...  # (ok, short text for the display)


def split_segments(points: list[np.ndarray | None]) -> list[np.ndarray]:
    """Polyline segments from a list of points with gaps (None)."""
    segments, cur = [], []
    for p in points + [None]:
        if p is not None:
            cur.append(p)
            continue
        if len(cur) > 1:
            segments.append(np.array(cur))
        cur = []
    return segments


class LiveViconSource:
    def __init__(self, vicon: ViconStream):
        self.vicon = vicon

    def pose_at(self, subject: str, t: float) -> np.ndarray | None:
        return self.vicon.pose_at(subject, t)

    def trail(self, subject: str, t: float, seconds: float, offset: np.ndarray) -> list[np.ndarray]:
        samples = self.vicon.track(subject, t - seconds, t)[::4]  # ~50 Hz is plenty for a line
        return split_segments([None if T is None else apply(T, offset[None])[0] for _, T in samples])

    def status(self) -> tuple[bool, str]:
        ok = self.vicon.state == State.STREAMING
        return ok, f"vicon {self.vicon.state.value}" + ("" if ok else f": {self.vicon.last_error}")


class RecordingSource:
    def __init__(self, rec: Recording, max_gap_s: float = 0.05):
        self.rec, self.max_gap_s = rec, max_gap_s

    def pose_at(self, subject: str, t: float) -> np.ndarray | None:
        if subject not in self.rec.tracks:
            return None
        T, ok = self.rec.pose_at(subject, [t], self.max_gap_s)
        return T[0] if ok[0] else None

    def trail(self, subject: str, t: float, seconds: float, offset: np.ndarray, rate_hz: float = 50.0):
        if subject not in self.rec.tracks:
            return []
        times = np.linspace(t - seconds, t, max(2, int(seconds * rate_hz)))
        T, ok = self.rec.pose_at(subject, times, self.max_gap_s)
        return split_segments([apply(Ti, offset[None])[0] if v else None for Ti, v in zip(T, ok)])

    def status(self) -> tuple[bool, str]:
        return True, f"replay {self.rec.path.name}"
