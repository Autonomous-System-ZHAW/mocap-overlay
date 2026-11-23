"""File writers for recordings: H.264 video through an ffmpeg pipe, and CSV logs."""

import csv
import subprocess
from pathlib import Path

import numpy as np

CODECS = {
    # quality: x264 CRF / NVENC CQ (lower = better; 18 is visually lossless for this content)
    "x264": lambda q: ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(q)],
    "nvenc": lambda q: ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(q), "-b:v", "0"],
}


class FFmpegVideoWriter:
    """Writes BGR frames to a video file. Use .mkv for recordings (survives a crash), .mp4 for output."""

    def __init__(self, path: str | Path, size: tuple[int, int], fps: float, codec: str = "x264", quality: int = 18,
                 extra_args: list[str] | None = None):
        self.size = size
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}", "-i", "-",
            *CODECS[codec](quality), *(extra_args or []), "-pix_fmt", "yuv420p",
            *(["-movflags", "+faststart"] if str(path).endswith(".mp4") else []), str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, image: np.ndarray) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError(f"frame size {image.shape[1]}x{image.shape[0]} != video size {self.size}")
        self._proc.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> int:
        self._proc.stdin.close()
        return self._proc.wait()


class CsvLog:
    """Append-only CSV. Convention for all logs of a recording: the first column is t_host
    (host clock, seconds since epoch), so streams from different sources can be aligned."""

    def __init__(self, path: str | Path, header: list[str], flush_every: int = 200):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self._n, self._flush_every = 0, flush_every

    def write(self, row: list) -> None:
        self._w.writerow(row)
        self._n += 1
        if self._n % self._flush_every == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()
