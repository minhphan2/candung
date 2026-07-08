"""
traffic_monitor.py — Hệ thống Phát hiện Ùn tắc Giao thông (All-in-One)

Gộp toàn bộ config.py, pipeline.py, visualization.py, main.py thành 1 file duy nhất.

Cách chạy:
    python traffic_monitor.py                           # Dùng mặc định
    python traffic_monitor.py --source traffic.mp4     # Chỉ định video
    python traffic_monitor.py --source 0               # Dùng webcam
    python traffic_monitor.py --model yolo11n.pt       # Chỉ định model

Phím tắt:
    Q → Thoát
    P → Tạm dừng / Tiếp tục
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 1: CẤU HÌNH (config.py)
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """
    Toàn bộ thông số cấu hình của hệ thống.
    Chỉnh sửa ROI_POLYGON để phù hợp với góc nhìn camera của bạn.
    """

    # ── Model & Input ─────────────────────────────────────────────────
    MODEL_PATH: str = "yolo11n.pt"
    VIDEO_SOURCE: str | int = "traffic.mp4"

    # ── Vùng quan sát (ROI) — đa giác lồi, tọa độ pixel ──────────────
    # Mặc định: hình thang phù hợp camera giao thông nhìn thẳng.
    # THAY thành tọa độ thực tế của camera bạn.
    ROI_POLYGON: np.ndarray = field(default_factory=lambda: np.array([
        [200, 400],
        [1080, 400],
        [1280, 700],
        [0, 700],
    ], dtype=np.int32))

    # ── COCO Vehicle Class IDs: 2=car, 3=moto, 5=bus, 7=truck ────────
    TARGET_CLASSES: list[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # ── Ngưỡng quyết định Ùn tắc ─────────────────────────────────────
    OCCUPANCY_HIGH: float = 0.30          # O >= this → mẫu = 1 (đông)
    BUFFER_SIZE: int = 20                 # Số slot buffer (mỗi slot = 1 giây)
    CONGESTION_MEAN_THRESHOLD: float = 0.6  # CM >= this → GRIDLOCK

    # ── Làm mịn thời gian (tính bằng giây) ───────────────────────────
    SLIDING_WINDOW_SECONDS: float = 1.5
    MOTION_AVG_SECONDS: float = 1.0

    # ── Phát hiện ─────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.35
    TRACKER_TYPE: str = "bytetrack.yaml"

    # ── Hiển thị ──────────────────────────────────────────────────────
    DISPLAY_WIDTH: int | None = None      # None = giữ độ phân giải gốc

    # ── Tên loại xe (dùng cho nhãn overlay) ──────────────────────────
    CLASS_NAMES: dict[int, str] = field(default_factory=lambda: {
        2: "Car",
        3: "Moto",
        5: "Bus",
        7: "Truck",
    })


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 2: LOGIC PHÁT HIỆN ÙN TẮC (pipeline.py)
# ═══════════════════════════════════════════════════════════════════════
class TrafficState(Enum):
    """
    Hai trạng thái giao thông: Ùn tắc hoặc Thông thoáng.
    Mỗi trạng thái chứa: nhãn tiếng Anh, tiếng Việt, emoji, màu BGR.
    """
    GRIDLOCK  = ("GRIDLOCK",  "Ùn tắc nặng",  "🚨", (0, 0, 200))
    FREE_FLOW = ("FREE FLOW", "Thông thoáng", "🟢", (0, 210, 0))

    def __init__(self, label: str, label_vi: str, emoji: str, color_bgr: tuple[int, int, int]):
        self.label = label
        self.label_vi = label_vi
        self.emoji = emoji
        self.color_bgr = color_bgr


class Metrics(NamedTuple):
    """Snapshot các chỉ số tính toán cho một frame."""
    occupancy: float        # O  ∈ [0.0, 1.0]
    motion_variance: float  # V  (đã làm mịn, px/s)
    state: TrafficState     # Kết quả quyết định
    vehicle_count: int      # Số xe trong ROI frame này
    buffer_cm: float        # Current Mean của buffer (0.0 – 1.0)


class CongestionPipeline:
    """
    Bộ máy phát hiện ùn tắc giao thông thời gian thực.

    Quy trình mỗi frame:
        1. Chạy YOLO tracking (giữ ID ổn định xuyên suốt video).
        2. Với mỗi xe trong ROI: tính diện tích giao nhau chính xác.
        3. Tổng hợp thành Tỷ lệ Chiếm dụng (Occupancy Ratio O).
        4. Cập nhật lịch sử centroid; tính Motion Variance V đã làm mịn.
        5. Lấy mẫu O mỗi 1 giây vào buffer vòng → tính Current Mean (CM).
        6. CM >= ngưỡng → GRIDLOCK, ngược lại → FREE FLOW.
    """

    def __init__(self, cfg: Config, fps: float = 30.0) -> None:
        self.cfg = cfg
        self.fps = max(fps, 1.0)

        # ── Load YOLO model ──────────────────────────────────────────
        self.model = YOLO(cfg.MODEL_PATH)

        # ── Tính diện tích ROI (công thức Shoelace qua OpenCV) ───────
        self.roi_polygon = cfg.ROI_POLYGON.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_area: float = cv2.contourArea(cfg.ROI_POLYGON.astype(np.float32))
        if self.roi_area < 1.0:
            raise ValueError("ROI polygon area is effectively zero — kiểm tra lại ROI_POLYGON.")

        # ── Chuyển đổi giây → số frame cho sliding windows ───────────
        sw_frames = max(int(cfg.SLIDING_WINDOW_SECONDS * self.fps), 1)
        ma_frames = max(int(cfg.MOTION_AVG_SECONDS * self.fps), 1)

        # ── Lịch sử centroid: track_id → deque of (cx, cy) ───────────
        self._centroid_history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=sw_frames)
        )

        # ── Lịch sử V (cho bộ lọc Moving Average) ────────────────────
        self._v_history: deque[float] = deque(maxlen=ma_frames)

        # ── Buffer 20 slot + Running Mean ─────────────────────────────
        self._buffer: list[int] = [0] * cfg.BUFFER_SIZE
        self._buffer_idx: int = 0
        self._buffer_cm: float = 0.0
        self._frames_per_second: int = max(int(self.fps), 1)
        self._frame_counter: int = 0
        self._last_occupancy: float = 0.0

        # ── Chỉ số mới nhất (cho visualizer) ─────────────────────────
        self.latest_metrics: Metrics = Metrics(0.0, 0.0, TrafficState.FREE_FLOW, 0, 0.0)

        # ── ID track đang hoạt động (để dọn track cũ) ─────────────────
        self._active_ids: set[int] = set()

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Metrics, list]:
        """
        Chạy toàn bộ pipeline trên một frame BGR.
        Trả về: (frame gốc không sửa đổi, metrics, danh sách detections)
        """
        # 1. YOLO Tracking
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.cfg.TRACKER_TYPE,
            conf=self.cfg.CONFIDENCE_THRESHOLD,
            classes=self.cfg.TARGET_CLASSES,
            verbose=False,
        )
        detections = self._parse_results(results)

        # 2. Tính Occupancy Ratio O
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

        # 3. Cập nhật lịch sử centroid & tính Motion Variance V
        current_ids: set[int] = set()
        frame_displacements: list[float] = []
        for det in roi_detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            current_ids.add(tid)
            history = self._centroid_history[tid]
            if len(history) > 0:
                px, py = history[-1]
                disp = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                frame_displacements.append(disp)
            history.append((cx, cy))

        # Dọn các track đã biến mất
        stale_ids = self._active_ids - current_ids
        for sid in stale_ids:
            self._centroid_history.pop(sid, None)
        self._active_ids = current_ids

        raw_v = float(np.mean(frame_displacements)) if frame_displacements else 0.0
        self._v_history.append(raw_v)
        smoothed_v = float(np.mean(self._v_history)) if self._v_history else 0.0

        # 4. Lấy mẫu buffer mỗi 1 giây + Running Mean
        self._last_occupancy = occupancy
        self._frame_counter += 1
        if self._frame_counter >= self._frames_per_second:
            self._frame_counter = 0
            self._update_buffer(occupancy)

        state = self._decide_state()
        metrics = Metrics(
            occupancy=occupancy,
            motion_variance=smoothed_v,
            state=state,
            vehicle_count=len(roi_detections),
            buffer_cm=self._buffer_cm,
        )
        self.latest_metrics = metrics
        return frame, metrics, detections

    def _parse_results(self, results) -> list[dict]:
        detections: list[dict] = []
        if results is None or len(results) == 0:
            return detections
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        boxes = result.boxes
        if boxes.id is None:
            return detections
        ids = boxes.id.cpu().numpy().astype(int)
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for tid, xyxy, conf, cls_id in zip(ids, xyxys, confs, clss):
            detections.append({
                "track_id": int(tid),
                "bbox_xyxy": xyxy.tolist(),
                "cls_id": int(cls_id),
                "conf": float(conf),
            })
        return detections

    def _compute_intersection_area(self, bbox_xyxy: list[float]) -> float:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_poly = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2],
        ], dtype=np.float32).reshape((-1, 1, 2))
        ret, _ = cv2.intersectConvexConvex(self.roi_polygon, bbox_poly)
        return max(ret, 0.0)

    def _update_buffer(self, occupancy: float) -> None:
        """
        Mỗi 1 giây lấy mẫu O → 0 hoặc 1.
        Cập nhật buffer vòng + Running Mean:
            CM_new = CM + (a_new - a_old) / BUFFER_SIZE
        """
        step = self.cfg.BUFFER_SIZE
        a_new = 1 if occupancy >= self.cfg.OCCUPANCY_HIGH else 0
        a_old = self._buffer[self._buffer_idx]
        self._buffer[self._buffer_idx] = a_new
        self._buffer_idx = (self._buffer_idx + 1) % step
        self._buffer_cm = self._buffer_cm + (a_new - a_old) / step
        self._buffer_cm = max(0.0, min(1.0, self._buffer_cm))

    def _decide_state(self) -> TrafficState:
        if self._buffer_cm >= self.cfg.CONGESTION_MEAN_THRESHOLD:
            return TrafficState.GRIDLOCK
        return TrafficState.FREE_FLOW


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 3: VẼ GIAO DIỆN (visualization.py)
# ═══════════════════════════════════════════════════════════════════════
class Visualization:
    """Vẽ overlay lên frame: ROI, BBox xe, Dashboard HUD."""

    HUD_X, HUD_Y = 16, 16
    HUD_W, HUD_H = 380, 248
    HUD_BG_ALPHA = 0.72
    HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
    HUD_FONT_SCALE_TITLE = 0.70
    HUD_FONT_SCALE_BODY  = 0.60
    HUD_FONT_THICKNESS   = 2
    HUD_LINE_SPACING     = 32

    BBOX_COLORS: dict[int, tuple[int, int, int]] = {
        2: (255, 180, 50),   # Car
        3: (50, 255, 200),   # Motorcycle
        5: (255, 100, 255),  # Bus
        7: (100, 200, 255),  # Truck
    }
    DEFAULT_BBOX_COLOR = (200, 200, 200)

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def draw(self, frame: np.ndarray, metrics: Metrics, detections: list[dict]) -> np.ndarray:
        self._draw_roi(frame, metrics.state)
        self._draw_bboxes(frame, detections)
        self._draw_hud(frame, metrics)
        return frame

    def _draw_roi(self, frame: np.ndarray, state: TrafficState) -> None:
        overlay = frame.copy()
        color = state.color_bgr
        pts = self.cfg.ROI_POLYGON.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [self.cfg.ROI_POLYGON], color)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)

    def _draw_bboxes(self, frame: np.ndarray, detections: list[dict]) -> None:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            tid    = det["track_id"]
            cls_id = det["cls_id"]
            conf   = det["conf"]
            in_roi = det.get("in_roi", False)
            color     = self.BBOX_COLORS.get(cls_id, self.DEFAULT_BBOX_COLOR)
            thickness = 3 if in_roi else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cls_name = self.cfg.CLASS_NAMES.get(cls_id, "?")
            label = f"ID:{tid} {cls_name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, self.HUD_FONT, 0.45, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 6, label_y + baseline), color, -1)
            cv2.putText(frame, label, (x1 + 3, label_y - 2), self.HUD_FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, metrics: Metrics) -> None:
        x, y, w, h = self.HUD_X, self.HUD_Y, self.HUD_W, self.HUD_H

        # Nền bán trong suốt
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.HUD_BG_ALPHA, frame, 1 - self.HUD_BG_ALPHA, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), metrics.state.color_bgr, 2)

        ty = y + 28
        cv2.putText(frame, "TRAFFIC MONITOR", (x + 10, ty), self.HUD_FONT, self.HUD_FONT_SCALE_TITLE, (255, 255, 255), self.HUD_FONT_THICKNESS, cv2.LINE_AA)
        ty += 10
        cv2.line(frame, (x + 10, ty), (x + w - 10, ty), (100, 100, 100), 1)

        ty += self.HUD_LINE_SPACING
        cv2.putText(frame, f"State: {metrics.state.label}", (x + 10, ty), self.HUD_FONT, self.HUD_FONT_SCALE_BODY, metrics.state.color_bgr, self.HUD_FONT_THICKNESS, cv2.LINE_AA)
        ty += 22
        cv2.putText(frame, f"       ({metrics.state.label_vi})", (x + 10, ty), self.HUD_FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        ty += self.HUD_LINE_SPACING
        occ_pct = metrics.occupancy * 100
        cv2.putText(frame, f"Occupancy: {occ_pct:5.1f}%", (x + 10, ty), self.HUD_FONT, self.HUD_FONT_SCALE_BODY, (255, 255, 255), 1, cv2.LINE_AA)
        bar_x, bar_w, bar_h, bar_y = x + 220, 140, 14, ty - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * min(metrics.occupancy, 1.0)), bar_y + bar_h), metrics.state.color_bgr, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        ty += self.HUD_LINE_SPACING
        cm, cm_thresh = metrics.buffer_cm, self.cfg.CONGESTION_MEAN_THRESHOLD
        cv2.putText(frame, f"CM: {cm:.2f} / {cm_thresh:.2f}", (x + 10, ty), self.HUD_FONT, self.HUD_FONT_SCALE_BODY, (255, 255, 255), 1, cv2.LINE_AA)
        bar_x, bar_w, bar_h, bar_y = x + 220, 140, 14, ty - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cm_color = (0, 0, 200) if cm >= cm_thresh else ((0, 200, 255) if cm >= cm_thresh * 0.5 else (0, 200, 0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * min(cm, 1.0)), bar_y + bar_h), cm_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        ty += self.HUD_LINE_SPACING
        cv2.putText(frame, f"Vehicles in ROI: {metrics.vehicle_count}", (x + 10, ty), self.HUD_FONT, self.HUD_FONT_SCALE_BODY, (0, 220, 255), self.HUD_FONT_THICKNESS, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 4: VÒNG LẶP CHÍNH (main.py)
# ═══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hệ thống Phát hiện Ùn tắc Giao thông")
    parser.add_argument("--source",      type=str,   default=None)
    parser.add_argument("--model",       type=str,   default=None)
    parser.add_argument("--occ-thresh",  type=float, default=None)
    parser.add_argument("--buffer-size", type=int,   default=None)
    parser.add_argument("--cm-thresh",   type=float, default=None)
    parser.add_argument("--roi",         type=str,   default=None,
                        help="Tọa độ đa giác ROI: 'x1,y1,x2,y2,...' (ít nhất 3 đỉnh)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Chạy không hiển thị cửa sổ (headless)")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.source is not None:
        try:    cfg.VIDEO_SOURCE = int(args.source)
        except: cfg.VIDEO_SOURCE = args.source
    if args.model       is not None: cfg.MODEL_PATH = args.model
    if args.occ_thresh  is not None: cfg.OCCUPANCY_HIGH = args.occ_thresh
    if args.buffer_size is not None: cfg.BUFFER_SIZE = args.buffer_size
    if args.cm_thresh   is not None: cfg.CONGESTION_MEAN_THRESHOLD = args.cm_thresh
    if args.roi is not None:
        coords = list(map(int, args.roi.split(",")))
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError("--roi cần ít nhất 3 đỉnh: 'x1,y1,x2,y2,x3,y3,...'")
        cfg.ROI_POLYGON = np.array(coords, dtype=np.int32).reshape(-1, 2)
    return cfg


def main() -> None:
    args = parse_args()
    cfg  = build_config(args)

    print("=" * 60)
    print("  HỆ THỐNG PHÁT HIỆN ÙN TẮC GIAO THÔNG")
    print("=" * 60)
    print(f"  Model     : {cfg.MODEL_PATH}")
    print(f"  Nguồn     : {cfg.VIDEO_SOURCE}")
    print(f"  ROI       : {len(cfg.ROI_POLYGON)} đỉnh")
    print(f"  Occ thresh: {cfg.OCCUPANCY_HIGH}")
    print(f"  Buffer    : {cfg.BUFFER_SIZE} slot x 1s")
    print(f"  CM thresh : {cfg.CONGESTION_MEAN_THRESHOLD}")
    print("=" * 60)

    cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được nguồn video: {cfg.VIDEO_SOURCE}")
        sys.exit(1)

    fps_video   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline    = CongestionPipeline(cfg, fps=fps_video)
    viz         = Visualization(cfg)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {width}x{height} @ {fps_video:.1f} FPS, {total_frames} frames")
    print("-" * 60)

    paused    = False
    frame_idx = 0

    try:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("p"): paused = False
                elif key == ord("q"): break
                continue

            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Hết video.")
                break

            if cfg.DISPLAY_WIDTH is not None:
                scale = cfg.DISPLAY_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            frame, metrics, detections = pipeline.process_frame(frame)
            annotated = viz.draw(frame, metrics, detections)

            # FPS (góc dưới phải)
            elapsed     = time.perf_counter() - t_start
            fps_actual  = 1.0 / max(elapsed, 1e-6)
            h, w        = annotated.shape[:2]
            cv2.putText(annotated, f"FPS: {fps_actual:.1f}", (w - 160, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # Thời gian (góc dưới phải phía trên FPS)
            frame_idx  += 1
            elapsed_sec = frame_idx / fps_video
            time_info   = f"Time: {elapsed_sec:.1f}s"
            if total_frames > 0:
                time_info += f" / {total_frames / fps_video:.1f}s"
            cv2.putText(annotated, time_info, (w - 250, h - 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            # Console log mỗi 1 giây
            log_interval = max(int(fps_video), 1)
            if frame_idx % log_interval == 0:
                buf_str = "".join(str(b) for b in pipeline._buffer)
                print(
                    f"  [{elapsed_sec:>7.1f}s] "
                    f"O={metrics.occupancy:.3f}  "
                    f"CM={metrics.buffer_cm:.2f}  "
                    f"Buf=[{buf_str}]  "
                    f"State={metrics.state.label:<12s}  "
                    f"x{metrics.vehicle_count}"
                )

            if not args.no_display:
                cv2.imshow("Traffic Congestion Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):   break
                elif key == ord("p"): paused = True

    except KeyboardInterrupt:
        print("\n[INFO] Người dùng dừng chương trình.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  Đã xử lý {frame_idx} frames. Thoát.")


if __name__ == "__main__":
    main()
