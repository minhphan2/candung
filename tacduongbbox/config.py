"""
config.py — Centralized Configuration for Traffic Congestion Detection Pipeline.

All tunable constants, thresholds, and model paths are stored here.
Adjust ROI_POLYGON to match your camera perspective.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    """
    Stores all pipeline configuration constants.

    Attributes:
        MODEL_PATH: Path to the YOLOv8/YOLO11 model weights.
        VIDEO_SOURCE: Path to a video file, or 0 for webcam.
        ROI_POLYGON: Numpy array of (N, 2) vertices defining the Region of Interest.
                     Coordinates are in pixel space of the input frame.
                     Must be a CONVEX polygon for cv2.intersectConvexConvex.
        TARGET_CLASSES: COCO class IDs for vehicles we care about.
                        2=car, 3=motorcycle, 5=bus, 7=truck
        OCCUPANCY_HIGH: Threshold for "high" area occupancy ratio (0.0–1.0).
        MOTION_LOW: Threshold for "low" average motion (pixels/frame).
                    Below this, vehicles are considered nearly stationary.
        SLIDING_WINDOW_SIZE: Number of frames to keep in centroid history
                             for motion variance calculation (~30 frames ≈ 1s at 30fps).
        MOTION_AVG_WINDOW: Window size for the moving-average filter on V
                           to smooth out frame-to-frame jitter.
        CONFIDENCE_THRESHOLD: Minimum detection confidence to accept a YOLO result.
        TRACKER_TYPE: Tracker backend for YOLO .track() — "bytetrack" or "botsort".
        DISPLAY_WIDTH: Resize output frame width for display (None = original).
    """

    # ── Model & Input ────────────────────────────────────────────────
    MODEL_PATH: str = "yolo11n.pt"
    VIDEO_SOURCE: str | int = "traffic.mp4"

    # ── Region of Interest (convex polygon, pixel coordinates) ──────
    # Default: a trapezoidal ROI typical of a forward-facing traffic cam.
    # **REPLACE** these with actual coordinates for your camera view.
    ROI_POLYGON: np.ndarray = field(default_factory=lambda: np.array([
        [200, 400],
        [1080, 400],
        [1280, 700],
        [0, 700],
    ], dtype=np.int32))

    # ── COCO Vehicle Class IDs ───────────────────────────────────────
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    TARGET_CLASSES: list[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # ── Congestion Decision Thresholds ───────────────────────────────
    OCCUPANCY_HIGH: float = 0.30          # O >= this → mẫu = 1 (đông), ngược lại = 0
    BUFFER_SIZE: int = 20                 # Số slot buffer (mỗi slot = 1 giây)
    CONGESTION_MEAN_THRESHOLD: float = 0.6  # CM >= this → GRIDLOCK (60% = 12/20 giây bị đông)

    # ── Temporal Smoothing (tính bằng giây) ──────────────────────────
    SLIDING_WINDOW_SECONDS: float = 1.5   # lịch sử centroid mỗi track (giây)
    MOTION_AVG_SECONDS: float = 1.0       # cửa sổ moving-average cho V (giây)

    # ── Detection ────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.35
    TRACKER_TYPE: str = "bytetrack.yaml"

    # ── Display ──────────────────────────────────────────────────────
    DISPLAY_WIDTH: int | None = None      # None → keep original resolution

    # ── Class Name Mapping (for overlay labels) ──────────────────────
    CLASS_NAMES: dict[int, str] = field(default_factory=lambda: {
        2: "Car",
        3: "Moto",
        5: "Bus",
        7: "Truck",
    })
