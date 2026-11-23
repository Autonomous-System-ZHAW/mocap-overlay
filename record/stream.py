"""Resolve and open video sources (RTSP URL from camera.yaml, device index or video file)."""

import os
import re

import cv2

# FFmpeg options for RTSP: TCP avoids broken frames from lost UDP packets; no input buffering.
RTSP_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


def redact(url) -> str:
    """Hide user:password in a URL for printing."""
    return re.sub(r"//[^/@]*@", "//***@", str(url))


def resolve_source(camera_cfg: dict, override: str | None = None) -> str | int:
    """`override` if given, else the camera's rtsp_url with ${ENV} credentials expanded."""
    source = override or os.path.expandvars(camera_cfg["rtsp_url"])
    if "${" in str(source):
        raise SystemExit(f"unexpanded variable in stream URL, set the env vars: {source}")
    return int(source) if str(source).isdigit() else source


def is_network_stream(source) -> bool:
    return isinstance(source, str) and "://" in source


def is_file(source) -> bool:
    return isinstance(source, str) and not is_network_stream(source)


def try_open(source) -> cv2.VideoCapture | None:
    """Open a source once. Returns None on failure (e.g. 503 while the camera still holds an old session)."""
    if is_network_stream(source):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_OPTIONS
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    return cap
