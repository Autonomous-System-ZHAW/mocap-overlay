"""Minimal ctypes binding of the Vicon DataStream SDK C API (libViconDataStreamSDK_C.so).

Only what this project needs: connect, fetch frames, rigid body poses and markers.
Layouts follow CTypeDefs.h of SDK 1.13 (structs are packed with #pragma pack(1)).
Units are converted to metres; poses are returned as 4x4 T_world_<segment>.
"""

import ctypes as C
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUCCESS = 2  # CResult.Success
RESULT_NAMES = [
    "UnknownResult", "NotImplemented", "Success", "InvalidHostName", "InvalidMulticastIP",
    "ClientAlreadyConnected", "ClientConnectionFailed", "ServerAlreadyTransmittingMulticast",
    "ServerNotTransmittingMulticast", "NotConnected", "NoFrame", "InvalidIndex", "InvalidCameraName",
    "InvalidSubjectName", "InvalidSegmentName", "InvalidMarkerName", "InvalidDeviceName",
    "InvalidDeviceOutputName", "InvalidLatencySampleName", "CoLinearAxes", "LeftHandedAxes",
    "HapticAlreadySet", "EarlyDataRequested", "LateDataRequested", "InvalidOperation", "NotSupported",
    "ConfigurationFailed", "NotPresent",
]
STREAM_MODES = {"ClientPull": 0, "ClientPullPreFetch": 1, "ServerPush": 2}
MM_TO_M = 1e-3


def _struct(name: str, fields: list) -> type:
    # #pragma pack(1): no padding. ctypes only supports _pack_ with the "ms" layout, which is
    # identical to GCC for these flat int/double structs (sizes checked against CTypeDefs.h).
    return type(name, (C.Structure,), {"_pack_": 1, "_layout_": "ms", "_fields_": [("Result", C.c_int), *fields]})


OutUInt = _struct("OutUInt", [("Value", C.c_uint)])  # FrameNumber, SubjectCount, MarkerCount, ...
OutDouble = _struct("OutDouble", [("Value", C.c_double)])  # FrameRate, LatencyTotal
OutTranslation = _struct("OutTranslation", [("Translation", C.c_double * 3), ("Occluded", C.c_int)])
OutRotMatrix = _struct("OutRotMatrix", [("Rotation", C.c_double * 9), ("Occluded", C.c_int)])
OutQuaternion = _struct("OutQuaternion", [("Rotation", C.c_double * 4), ("Occluded", C.c_int)])
OutUnlabeled = _struct("OutUnlabeled", [("Translation", C.c_double * 3), ("MarkerID", C.c_uint)])


class ViconError(RuntimeError):
    pass


def _check(result: int, what: str) -> None:
    if result != SUCCESS:
        name = RESULT_NAMES[result] if 0 <= result < len(RESULT_NAMES) else str(result)
        raise ViconError(f"{what}: {name}")


@dataclass
class SegmentPose:
    T_world_segment: np.ndarray  # (4, 4), metres
    occluded: bool


class ViconClient:
    """Thin, synchronous wrapper. Not thread safe; use one client per thread."""

    def __init__(self, sdk_dir: str | Path):
        lib_path = Path(sdk_dir) / "libViconDataStreamSDK_C.so"
        if not lib_path.exists():
            raise FileNotFoundError(f"{lib_path} not found, copy the SDK's Linux64 .so files to {sdk_dir}")
        self._lib = lib = C.CDLL(str(lib_path.resolve()))
        cp, s, i, u = C.c_void_p, C.c_char_p, C.c_int, C.c_uint
        sig = {
            "Client_Create": ([], cp),
            "Client_Destroy": ([cp], None),
            "Client_Connect": ([cp, s], i),
            "Client_Disconnect": ([cp], i),
            "Client_IsConnected": ([cp], i),
            "Client_EnableSegmentData": ([cp], i),
            "Client_EnableMarkerData": ([cp], i),
            "Client_EnableUnlabeledMarkerData": ([cp], i),
            "Client_SetStreamMode": ([cp, i], i),
            "Client_SetBufferSize": ([cp, u], None),
            "Client_GetFrame": ([cp], i),
            "Client_GetFrameNumber": ([cp, C.POINTER(OutUInt)], None),
            "Client_GetFrameRate": ([cp, C.POINTER(OutDouble)], None),
            "Client_GetLatencyTotal": ([cp, C.POINTER(OutDouble)], None),
            "Client_GetSubjectCount": ([cp, C.POINTER(OutUInt)], None),
            "Client_GetSubjectName": ([cp, u, i, s], i),
            "Client_GetSubjectRootSegmentName": ([cp, s, i, s], i),
            "Client_GetSegmentGlobalTranslation": ([cp, s, s, C.POINTER(OutTranslation)], None),
            "Client_GetSegmentGlobalRotationMatrix": ([cp, s, s, C.POINTER(OutRotMatrix)], None),
            "Client_GetSegmentGlobalRotationQuaternion": ([cp, s, s, C.POINTER(OutQuaternion)], None),
            "Client_GetMarkerCount": ([cp, s, C.POINTER(OutUInt)], None),
            "Client_GetMarkerName": ([cp, s, u, i, s], i),
            "Client_GetMarkerGlobalTranslation": ([cp, s, s, C.POINTER(OutTranslation)], None),
            "Client_GetUnlabeledMarkerCount": ([cp, C.POINTER(OutUInt)], None),
            "Client_GetUnlabeledMarkerGlobalTranslation": ([cp, u, C.POINTER(OutUnlabeled)], None),
        }
        for name, (args, res) in sig.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
        self._c = lib.Client_Create()

    # --- connection -------------------------------------------------------------
    def connect(self, host: str, stream_mode: str = "ServerPush") -> None:
        _check(self._lib.Client_Connect(self._c, host.encode()), f"connect {host}")
        _check(self._lib.Client_EnableSegmentData(self._c), "enable segment data")
        _check(self._lib.Client_EnableMarkerData(self._c), "enable marker data")
        _check(self._lib.Client_EnableUnlabeledMarkerData(self._c), "enable unlabeled marker data")
        _check(self._lib.Client_SetStreamMode(self._c, STREAM_MODES[stream_mode]), "set stream mode")

    def is_connected(self) -> bool:
        return bool(self._lib.Client_IsConnected(self._c))

    def disconnect(self) -> None:
        if self.is_connected():
            self._lib.Client_Disconnect(self._c)

    def close(self) -> None:
        if self._c:
            self.disconnect()
            self._lib.Client_Destroy(self._c)
            self._c = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- frames -----------------------------------------------------------------
    def get_frame(self) -> None:
        """Blocks until the next frame (ServerPush) and makes it current."""
        _check(self._lib.Client_GetFrame(self._c), "get frame")

    def _uint(self, fn, *args) -> int:
        out = OutUInt()
        fn(self._c, *args, C.byref(out))
        _check(out.Result, fn.__name__)
        return out.Value

    def _double(self, fn) -> float:
        out = OutDouble()
        fn(self._c, C.byref(out))
        _check(out.Result, fn.__name__)
        return out.Value

    def frame_number(self) -> int:
        return self._uint(self._lib.Client_GetFrameNumber)

    def frame_rate(self) -> float:
        return self._double(self._lib.Client_GetFrameRate)

    def latency_s(self) -> float:
        """Total latency from camera exposure to the SDK receiving the frame [s]."""
        return self._double(self._lib.Client_GetLatencyTotal)

    # --- subjects ---------------------------------------------------------------
    def _string(self, fn, *args) -> str:
        buf = C.create_string_buffer(256)
        _check(fn(self._c, *args, len(buf), buf), fn.__name__)
        return buf.value.decode()

    def subjects(self) -> list[str]:
        n = self._uint(self._lib.Client_GetSubjectCount)
        return [self._string(self._lib.Client_GetSubjectName, i) for i in range(n)]

    def root_segment(self, subject: str) -> str:
        return self._string(self._lib.Client_GetSubjectRootSegmentName, subject.encode())

    def segment_pose(self, subject: str, segment: str | None = None) -> SegmentPose:
        """Global pose of a segment (default: the root segment) in the Vicon world."""
        segment = segment or self.root_segment(subject)
        s, g = subject.encode(), segment.encode()
        t, r = OutTranslation(), OutRotMatrix()
        self._lib.Client_GetSegmentGlobalTranslation(self._c, s, g, C.byref(t))
        _check(t.Result, f"translation {subject}/{segment}")
        self._lib.Client_GetSegmentGlobalRotationMatrix(self._c, s, g, C.byref(r))
        _check(r.Result, f"rotation {subject}/{segment}")
        T = np.eye(4)
        T[:3, :3] = np.array(r.Rotation).reshape(3, 3)  # row-major
        T[:3, 3] = np.array(t.Translation) * MM_TO_M
        return SegmentPose(T, bool(t.Occluded or r.Occluded))

    def segment_quaternion(self, subject: str, segment: str | None = None) -> np.ndarray:
        """Raw (x, y, z, w) quaternion, used to cross-check the matrix layout."""
        segment = segment or self.root_segment(subject)
        q = OutQuaternion()
        self._lib.Client_GetSegmentGlobalRotationQuaternion(self._c, subject.encode(), segment.encode(), C.byref(q))
        _check(q.Result, f"quaternion {subject}/{segment}")
        return np.array(q.Rotation)

    def markers(self, subject: str) -> dict[str, np.ndarray | None]:
        """Labelled markers of a subject in world coordinates [m]; None if occluded."""
        n = self._uint(self._lib.Client_GetMarkerCount, subject.encode())
        out = {}
        for i in range(n):
            name = self._string(self._lib.Client_GetMarkerName, subject.encode(), i)
            t = OutTranslation()
            self._lib.Client_GetMarkerGlobalTranslation(self._c, subject.encode(), name.encode(), C.byref(t))
            _check(t.Result, f"marker {subject}/{name}")
            out[name] = None if t.Occluded else np.array(t.Translation) * MM_TO_M
        return out

    def unlabeled_markers(self) -> np.ndarray:
        """(N, 3) unlabelled marker positions [m]."""
        n = self._uint(self._lib.Client_GetUnlabeledMarkerCount)
        pts = []
        for i in range(n):
            t = OutUnlabeled()
            self._lib.Client_GetUnlabeledMarkerGlobalTranslation(self._c, i, C.byref(t))
            _check(t.Result, "unlabeled marker")
            pts.append(np.array(t.Translation) * MM_TO_M)
        return np.array(pts).reshape(-1, 3)


def quat_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw / np.linalg.norm(q_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
