"""
pipeline.py — Core Congestion Detection Pipeline.

Encapsulates YOLO tracking, geometric intersection computation,
occupancy ratio (O), motion variance (V), and the 2D decision matrix.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from enum import Enum
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


# ═══════════════════════════════════════════════════════════════════════
# Traffic State Enumeration
# ═══════════════════════════════════════════════════════════════════════
class TrafficState(Enum):
    """
    Two traffic states: Congested or Free.

    Each state carries:
        - label : English display name
        - label_vi : Vietnamese display name
        - emoji : visual indicator
        - color_bgr : BGR color tuple for ROI overlay
    """
    GRIDLOCK   = ("GRIDLOCK",   "Ùn tắc nặng", "🚨", (0, 0, 200))    # Đỏ đậm
    FREE_FLOW  = ("FREE FLOW",  "Thông thoáng", "🟢", (0, 210, 0))    # Xanh lá

    def __init__(self, label: str, label_vi: str, emoji: str, color_bgr: tuple[int, int, int]):
        self.label = label
        self.label_vi = label_vi
        self.emoji = emoji
        self.color_bgr = color_bgr


# ═══════════════════════════════════════════════════════════════════════
# Named tuple for per-frame metrics snapshot
# ═══════════════════════════════════════════════════════════════════════
class Metrics(NamedTuple):
    """Snapshot of computed metrics for a single frame."""
    occupancy: float           # O  ∈ [0.0, 1.0]
    motion_variance: float     # V  (smoothed, px/s)
    state: TrafficState        # Decision result
    vehicle_count: int         # Vehicles inside ROI this frame
    buffer_cm: float           # Current Mean của buffer (0.0 – 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Congestion Pipeline
# ═══════════════════════════════════════════════════════════════════════
class CongestionPipeline:
    """
    Real-time traffic congestion detector.

    Workflow per frame:
        1. Run YOLO tracking with persistent IDs.
        2. For each detected vehicle bbox intersecting the ROI,
           compute exact geometric intersection area.
        3. Aggregate into Occupancy Ratio O.
        4. Update per-ID centroid history; compute smoothed Motion Variance V.
        5. Map (O, V) → TrafficState via the 2D decision matrix.
    """

    def __init__(self, cfg: Config, fps: float = 30.0) -> None:
        """
        Initialize the pipeline.

        Args:
            cfg: Pipeline configuration dataclass.
            fps: Video FPS — dùng để chuyển đổi giây → frame nội bộ.
        """
        self.cfg = cfg
        self.fps = max(fps, 1.0)

        # ── Load YOLO model ──────────────────────────────────────────
        self.model = YOLO(cfg.MODEL_PATH)

        # ── Precompute ROI area (Shoelace formula via OpenCV) ────────
        self.roi_polygon = cfg.ROI_POLYGON.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_area: float = cv2.contourArea(cfg.ROI_POLYGON.astype(np.float32))
        if self.roi_area < 1.0:
            raise ValueError("ROI polygon area is effectively zero — check ROI_POLYGON coordinates.")

        # ── Chuyển đổi giây → số frame cho sliding windows ───────────
        sw_frames = max(int(cfg.SLIDING_WINDOW_SECONDS * self.fps), 1)
        ma_frames = max(int(cfg.MOTION_AVG_SECONDS * self.fps), 1)

        # ── Centroid history: track_id → deque of (cx, cy) ───────────
        self._centroid_history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=sw_frames)
        )

        # ── Smoothed V history (for moving-average filter) ───────────
        self._v_history: deque[float] = deque(maxlen=ma_frames)

        # ── Buffer 20 slot + Running Mean (công thức update anh viết) ─
        self._buffer: list[int] = [0] * cfg.BUFFER_SIZE  # buffer[i] = 0 hoặc 1
        self._buffer_idx: int = 0              # vị trí ghi tiếp theo (circular)
        self._buffer_cm: float = 0.0           # Current Mean (running mean)
        self._frames_per_second: int = max(int(self.fps), 1)  # bao nhiêu frame = 1 giây
        self._frame_counter: int = 0           # đếm frame để biết khi nào đủ 1 giây
        self._last_occupancy: float = 0.0      # O mới nhất (để lấy mẫu mỗi giây)

        # ── Latest metrics (available for the visualizer) ────────────
        self.latest_metrics: Metrics = Metrics(0.0, 0.0, TrafficState.FREE_FLOW, 0, 0.0)

        # ── Track IDs seen this frame (for stale-track pruning) ──────
        self._active_ids: set[int] = set()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Metrics, list]:
        """
        Run the full pipeline on a single BGR frame.

        Args:
            frame: Input BGR image (H, W, 3).

        Returns:
            annotated_frame: The frame (unmodified — visualization is separate).
            metrics: Computed (O, V, state, count) for this frame.
            detections: List of dicts with keys
                        {track_id, bbox_xyxy, cls_id, conf, in_roi, intersection_area}
        """
        # ── 1. YOLO Tracking ────────────────────────────────────────
        results = self.model.track(
            source=frame,
            persist=True,                      # keep IDs across frames
            tracker=self.cfg.TRACKER_TYPE,
            conf=self.cfg.CONFIDENCE_THRESHOLD,
            classes=self.cfg.TARGET_CLASSES,
            verbose=False,
        )

        detections = self._parse_results(results)

        # ── 2. Compute Occupancy Ratio O ─────────────────────────────
        total_intersection_area = 0.0
        roi_detections: list[dict] = []

        for det in detections:
            inter_area = self._compute_intersection_area(det["bbox_xyxy"])
            det["intersection_area"] = inter_area
            det["in_roi"] = inter_area > 0.0
            if det["in_roi"]:
                total_intersection_area += inter_area
                roi_detections.append(det)

        occupancy = float(np.clip(total_intersection_area / self.roi_area, 0.0, 1.0))

        # ── 3. Update centroid history & compute Motion Variance V ───
        current_ids: set[int] = set()
        frame_displacements: list[float] = []

        for det in roi_detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            current_ids.add(tid)

            history = self._centroid_history[tid]

            # If we have a previous position, compute displacement
            if len(history) > 0:
                px, py = history[-1]
                disp = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                frame_displacements.append(disp)

            history.append((cx, cy))

        # Prune stale tracks (IDs that disappeared from the ROI)
        stale_ids = self._active_ids - current_ids
        for sid in stale_ids:
            self._centroid_history.pop(sid, None)
        self._active_ids = current_ids

        # Raw V: mean displacement across all active vehicles this frame
        raw_v = float(np.mean(frame_displacements)) if frame_displacements else 0.0
        self._v_history.append(raw_v)

        # Smoothed V: moving average over the V history window
        smoothed_v = float(np.mean(self._v_history)) if self._v_history else 0.0

        # ── 4. Buffer sampling (mỗi 1 giây) + Running Mean ────────────
        self._last_occupancy = occupancy
        self._frame_counter += 1

        if self._frame_counter >= self._frames_per_second:
            self._frame_counter = 0
            self._update_buffer(occupancy)

        # Xác định state từ CM hiện tại
        state = self._decide_state()

        # ── Package metrics ──────────────────────────────────────────
        metrics = Metrics(
            occupancy=occupancy,
            motion_variance=smoothed_v,
            state=state,
            vehicle_count=len(roi_detections),
            buffer_cm=self._buffer_cm,
        )
        self.latest_metrics = metrics

        return frame, metrics, detections

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────
    def _parse_results(self, results) -> list[dict]:
        """
        Extract tracked detections from YOLO results.

        Returns a list of dicts:
            track_id, bbox_xyxy (x1,y1,x2,y2), cls_id, conf
        """
        detections: list[dict] = []

        if results is None or len(results) == 0:
            return detections

        result = results[0]

        # If no boxes at all, return empty
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        # tracker may fail to assign IDs on some frames
        if boxes.id is None:
            return detections

        ids = boxes.id.cpu().numpy().astype(int)
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for tid, xyxy, conf, cls_id in zip(ids, xyxys, confs, clss):
            detections.append({
                "track_id": int(tid),
                "bbox_xyxy": xyxy.tolist(),      # [x1, y1, x2, y2]
                "cls_id": int(cls_id),
                "conf": float(conf),
            })

        return detections

    def _compute_intersection_area(self, bbox_xyxy: list[float]) -> float:
        """
        Compute the exact geometric intersection area between a bounding box
        and the ROI polygon using cv2.intersectConvexConvex.

        Args:
            bbox_xyxy: [x1, y1, x2, y2] bounding box coordinates.

        Returns:
            Intersection area in pixels² (0.0 if no overlap).
        """
        x1, y1, x2, y2 = bbox_xyxy

        # Convert bbox to a 4-vertex convex polygon (clockwise)
        bbox_poly = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32).reshape((-1, 1, 2))

        # cv2.intersectConvexConvex requires contours in (N, 1, 2) float32
        ret, _intersection_region = cv2.intersectConvexConvex(
            self.roi_polygon, bbox_poly
        )

        # ret = intersection area; negative if no intersection
        return max(ret, 0.0)

    def _update_buffer(self, occupancy: float) -> None:
        """
        Mỗi 1 giây gọi hàm này 1 lần.
        Lấy mẫu: O >= ngưỡng → a_new = 1, ngược lại → a_new = 0.
        Update buffer circular + running mean theo công thức:

            CM_new = CM + (a_new - a_old) / step

        Trong đó:
            step  = BUFFER_SIZE (20)
            a_new = giá trị mới (0 hoặc 1)
            a_old = giá trị cũ nhất bị đẩy ra khỏi buffer
            CM    = Current Mean hiện tại
        """
        step = self.cfg.BUFFER_SIZE
        a_new = 1 if occupancy >= self.cfg.OCCUPANCY_HIGH else 0

        # Lấy giá trị cũ tại vị trí sắp bị ghi đè
        a_old = self._buffer[self._buffer_idx]

        # Ghi giá trị mới vào buffer (circular)
        self._buffer[self._buffer_idx] = a_new
        self._buffer_idx = (self._buffer_idx + 1) % step

        # Update running mean theo công thức: CM_new = CM + (a_new - a_old) / step
        self._buffer_cm = self._buffer_cm + (a_new - a_old) / step

        # Clamp để tránh lỗi floating point
        self._buffer_cm = max(0.0, min(1.0, self._buffer_cm))

    def _decide_state(self) -> TrafficState:
        """
        Dựa vào Current Mean (CM) của buffer:
            CM >= CONGESTION_MEAN_THRESHOLD  →  🚨 GRIDLOCK
            CM <  CONGESTION_MEAN_THRESHOLD  →  🟢 FREE FLOW
        """
        if self._buffer_cm >= self.cfg.CONGESTION_MEAN_THRESHOLD:
            return TrafficState.GRIDLOCK
        else:
            return TrafficState.FREE_FLOW
