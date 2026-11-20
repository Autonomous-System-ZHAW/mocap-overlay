"""Open the camera stream (RTSP URL from camera.yaml, device index or video file)."""

import os
import re
import time
from contextlib import contextmanager

import cv2


def redact(url: str) -> str:
    """Hide user:password in a URL for printing."""
    return re.sub(r"//[^/@]*@", "//***@", str(url))


def open_capture(camera_cfg: dict, override: str | None = None, retry_s: float = 90.0) -> cv2.VideoCapture:
    """Open `override` if given, else the camera's rtsp_url with ${ENV} credentials expanded.

    Retries for `retry_s` seconds: after an unclean exit the Axis keeps the old RTSP
    session until its timeout and answers 503 in the meantime.
    """
    source = override or os.path.expandvars(camera_cfg["rtsp_url"])
    if "${" in str(source):
        raise SystemExit(f"unexpanded variable in stream URL, set the env vars: {source}")
    source = int(source) if str(source).isdigit() else source
    is_stream = isinstance(source, str) and "://" in source

    deadline = time.monotonic() + (retry_s if is_stream else 0)
    while True:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on RTSP
            return cap
        cap.release()
        if time.monotonic() >= deadline:
            raise SystemExit(f"cannot open {redact(source)}")
        print(f"cannot open {redact(source)}, retrying in 5 s (old session may still be held by the camera)")
        time.sleep(5)


@contextmanager
def capture(camera_cfg: dict, override: str | None = None):
    """Open the stream and always release it, also on Ctrl+C, so the camera gets a TEARDOWN."""
    cap = open_capture(camera_cfg, override)
    try:
        yield cap
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
