"""ChArUco board definition, rendering and detection."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class BoardConfig:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BoardConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def build(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            dictionary,
        )
        board.setLegacyPattern(self.legacy_pattern)
        return board


@dataclass
class Detection:
    """ChArUco corners found in one image."""

    corners: np.ndarray  # (N, 1, 2) float32, pixels
    ids: np.ndarray  # (N, 1) int32, ChArUco corner ids

    def __len__(self) -> int:
        return len(self.ids)


def render_board(board: cv2.aruco.CharucoBoard, px_per_m: float, margin_px: int = 0) -> np.ndarray:
    """Render the board as a grayscale image at the given resolution."""
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    size = (round(sx * sq * px_per_m) + 2 * margin_px, round(sy * sq * px_per_m) + 2 * margin_px)
    return board.generateImage(size, marginSize=margin_px, borderBits=1)


def make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def detect(detector: cv2.aruco.CharucoDetector, image: np.ndarray) -> Detection | None:
    """Detect ChArUco corners. Returns None if nothing was found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) == 0:
        return None
    # OpenCV 4.x returns (N, 1, 2) / (N, 1), 5.x returns (N, 2) / (N,). Normalise to the
    # 4.x layout, which the drawing and calibration functions interpret unambiguously.
    return Detection(
        corners=np.asarray(corners, np.float32).reshape(-1, 1, 2),
        ids=np.asarray(ids, np.int32).reshape(-1, 1),
    )


def object_points(board: cv2.aruco.CharucoBoard, det: Detection) -> np.ndarray:
    """3D board coordinates (board frame, metres) of the detected corners, (N, 3)."""
    return board.getChessboardCorners()[det.ids.ravel()].astype(np.float32)


def effective_min_corners(board: cv2.aruco.CharucoBoard, requested: int) -> int:
    """Cap the per-view corner minimum at what the board can show (at least 4 for a homography)."""
    total = len(board.getChessboardCorners())
    if requested > total:
        print(f"WARNING: min_corners_per_view={requested} but the board only has {total} corners; using {total}")
    return max(4, min(requested, total))
