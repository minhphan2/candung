"""
congestion_node.py — CVOS Python Function Node: Traffic Congestion Detection

Pipeline CVOS hoàn chỉnh:
    [Camera Source - built-in CVOS]
        │ Topic: camera_frame (Image)
        ▼
    [yolo11n.onnx — ONNX Import]
        │ Topic: vehicle_bbox (Bbox)
        ▼
    [congestion_node.py — Python Import]  ← FILE NÀY
        ├── Input 1 (image): camera_frame
        └── Input 2 (bbox) : vehicle_bbox
        │ Topic: annotated_frame (Image)
        ▼
    [Dashboard Live Preview]

Cách upload lên CVOS:
    1. Dashboard → Python Import → chọn file này
    2. CVOS tự detect hàm `congestion_node`
    3. Input Topic 1 (image): camera_frame   (từ Camera Source Node)
    4. Input Topic 2 (bbox) : vehicle_bbox   (từ YOLO ONNX Node)
    5. Output Topic         : annotated_frame
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import cv2
import numpy as np
from cvos.topic.types_lib import Image, Bbox


# ═══════════════════════════════════════════════════════════════════════
# CẤU HÌNH — Chỉnh sửa theo camera thực tế của bạn
# ═══════════════════════════════════════════════════════════════════════

# Vùng quan sát (ROI) — Thay bằng tọa độ thực của camera bạn
ROI_POLYGON = np.array([
    [200, 400],
    [1080, 400],
    [1280, 700],
    [0, 700],
], dtype=np.int32)

# COCO class IDs: 2=car, 3=moto, 5=bus, 7=truck
TARGET_CLASSES = {2, 3, 5, 7}
CLASS_NAMES    = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

# Ngưỡng quyết định
OCCUPANCY_HIGH           = 0.30   # O >= này → mẫu = 1 (đông)
BUFFER_SIZE              = 20     # Số slot buffer (mỗi slot = 1 giây)
CONGESTION_MEAN_THRESHOLD = 0.60  # CM >= này → GRIDLOCK

# Tần suất video (fps) — Chỉnh cho đúng camera
ASSUMED_FPS = 30.0

# Cửa sổ làm mịn
SW_FRAMES = max(int(1.5 * ASSUMED_FPS), 1)   # lịch sử centroid
MA_FRAMES = max(int(1.0 * ASSUMED_FPS), 1)   # moving-average V

# ═══════════════════════════════════════════════════════════════════════
# TRẠNG THÁI NỘI BỘ — Duy trì xuyên suốt các lần gọi hàm
# ═══════════════════════════════════════════════════════════════════════
_centroid_history: dict = defaultdict(lambda: deque(maxlen=SW_FRAMES))
_v_history: deque        = deque(maxlen=MA_FRAMES)
_buffer: list            = [0] * BUFFER_SIZE
_buffer_idx: int         = 0
_buffer_cm: float        = 0.0
_frame_counter: int      = 0
_active_ids: set         = set()

# Tính diện tích ROI một lần
_roi_polygon_f32 = ROI_POLYGON.reshape((-1, 1, 2)).astype(np.float32)
_roi_area: float = cv2.contourArea(ROI_POLYGON.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════
# HÀM TÍNH TOÁN NỘI BỘ
# ═══════════════════════════════════════════════════════════════════════

def _compute_intersection_area(x1, y1, x2, y2) -> float:
    """Tính diện tích giao nhau giữa BBox và ROI (Sutherland-Hodgman)."""
    bbox_poly = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32
    ).reshape((-1, 1, 2))
    ret, _ = cv2.intersectConvexConvex(_roi_polygon_f32, bbox_poly)
    return max(ret, 0.0)


def _update_buffer(occupancy: float) -> None:
    """Cập nhật buffer vòng + Running Mean mỗi 1 giây."""
    global _buffer, _buffer_idx, _buffer_cm
    a_new = 1 if occupancy >= OCCUPANCY_HIGH else 0
    a_old = _buffer[_buffer_idx]
    _buffer[_buffer_idx] = a_new
    _buffer_idx = (_buffer_idx + 1) % BUFFER_SIZE
    _buffer_cm = _buffer_cm + (a_new - a_old) / BUFFER_SIZE
    _buffer_cm = max(0.0, min(1.0, _buffer_cm))


def _decide_state() -> tuple[str, str, tuple]:
    """Trả về (label, label_vi, color_bgr) dựa trên CM hiện tại."""
    if _buffer_cm >= CONGESTION_MEAN_THRESHOLD:
        return "GRIDLOCK", "Ùn tắc nặng", (0, 0, 200)
    return "FREE FLOW", "Thông thoáng", (0, 210, 0)


# ═══════════════════════════════════════════════════════════════════════
# HÀM VẼ GIAO DIỆN
# ═══════════════════════════════════════════════════════════════════════
FONT = cv2.FONT_HERSHEY_SIMPLEX
BBOX_COLORS = {2: (255, 180, 50), 3: (50, 255, 200), 5: (255, 100, 255), 7: (100, 200, 255)}

def _draw_roi(frame, color):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [ROI_POLYGON], color)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [ROI_POLYGON.reshape((-1, 1, 2))], True, color, 3)


def _draw_bboxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        tid    = det["tid"]
        cls_id = det["cls_id"]
        conf   = det["conf"]
        in_roi = det["in_roi"]
        color     = BBOX_COLORS.get(cls_id, (200, 200, 200))
        thickness = 3 if in_roi else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"ID:{tid} {CLASS_NAMES.get(cls_id,'?')} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.45, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + bl), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_hud(frame, occupancy, vehicle_count, label, label_vi, color):
    x, y, w, h = 16, 16, 380, 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    ty = y + 28
    cv2.putText(frame, "TRAFFIC MONITOR", (x+10, ty), FONT, 0.70, (255,255,255), 2, cv2.LINE_AA)
    ty += 10; cv2.line(frame, (x+10, ty), (x+w-10, ty), (100,100,100), 1)

    ty += 32
    cv2.putText(frame, f"State: {label}", (x+10, ty), FONT, 0.60, color, 2, cv2.LINE_AA)
    ty += 22
    cv2.putText(frame, f"       ({label_vi})", (x+10, ty), FONT, 0.45, (180,180,180), 1, cv2.LINE_AA)

    ty += 32
    occ_pct = occupancy * 100
    cv2.putText(frame, f"Occupancy: {occ_pct:5.1f}%", (x+10, ty), FONT, 0.60, (255,255,255), 1, cv2.LINE_AA)
    bx, bw, bh, by2 = x+220, 140, 14, ty-12
    cv2.rectangle(frame, (bx, by2), (bx+bw, by2+bh), (80,80,80), -1)
    cv2.rectangle(frame, (bx, by2), (bx+int(bw*min(occupancy,1.0)), by2+bh), color, -1)
    cv2.rectangle(frame, (bx, by2), (bx+bw, by2+bh), (180,180,180), 1)

    ty += 32
    cv2.putText(frame, f"CM: {_buffer_cm:.2f} / {CONGESTION_MEAN_THRESHOLD:.2f}", (x+10, ty), FONT, 0.60, (255,255,255), 1, cv2.LINE_AA)
    bx, bw, bh, by2 = x+220, 140, 14, ty-12
    cv2.rectangle(frame, (bx, by2), (bx+bw, by2+bh), (80,80,80), -1)
    cm_color = (0,0,200) if _buffer_cm >= CONGESTION_MEAN_THRESHOLD else ((0,200,255) if _buffer_cm >= CONGESTION_MEAN_THRESHOLD*0.5 else (0,200,0))
    cv2.rectangle(frame, (bx, by2), (bx+int(bw*min(_buffer_cm,1.0)), by2+bh), cm_color, -1)
    cv2.rectangle(frame, (bx, by2), (bx+bw, by2+bh), (180,180,180), 1)

    ty += 32
    cv2.putText(frame, f"Vehicles in ROI: {vehicle_count}", (x+10, ty), FONT, 0.60, (0,220,255), 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
# CVOS NODE FUNCTION — Đây là hàm CVOS sẽ tự động detect và wrap thành Node
# ═══════════════════════════════════════════════════════════════════════
def congestion_node(image: Image, bbox: Bbox) -> Image:
    """
    Nhận frame camera + danh sách xe từ YOLO → Tính toán tình trạng ùn tắc → Trả về frame đã vẽ HUD.

    CVOS tự đồng bộ (SyncGroup) 2 Topic đầu vào theo seq_no trước khi gọi hàm này.

    Input:
        image : Image — Frame BGRfull từ camera (qua Camera Source Node)
        bbox  : Bbox  — Tọa độ xe phát hiện được (qua YOLO ONNX Node)
    Output:
        Image — Frame đã vẽ ROI overlay, BBox, HUD Dashboard
    """
    global _frame_counter, _active_ids

    frame = image.copy()

    # ── Parse Bbox output từ YOLO ────────────────────────────────────
    # bbox.boxes     : (N, 4) float32 — [x1, y1, x2, y2]
    # bbox.scores    : (N,)   float32
    # bbox.class_ids : (N,)   int32
    boxes     = bbox.boxes     if bbox.boxes     is not None else np.empty((0, 4))
    scores    = bbox.scores    if bbox.scores    is not None else np.empty((0,))
    class_ids = bbox.class_ids if bbox.class_ids is not None else np.empty((0,), dtype=int)

    # ── Tính Occupancy Ratio O ───────────────────────────────────────
    total_inter = 0.0
    detections  = []

    # YOLO ONNX không có track_id — dùng index tạm thời làm ID
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        if int(cls_id) not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        inter = _compute_intersection_area(x1, y1, x2, y2)
        in_roi = inter > 0.0
        if in_roi:
            total_inter += inter
        detections.append({
            "tid": i, "xyxy": (x1, y1, x2, y2),
            "cls_id": int(cls_id), "conf": float(score),
            "in_roi": in_roi,
        })

    if _roi_area > 0:
        occupancy = float(np.clip(total_inter / _roi_area, 0.0, 1.0))
    else:
        occupancy = 0.0

    # ── Tính Motion Variance V ───────────────────────────────────────
    current_ids: set = set()
    displacements: list = []

    for det in [d for d in detections if d["in_roi"]]:
        tid = det["tid"]
        x1, y1, x2, y2 = det["xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        current_ids.add(tid)
        history = _centroid_history[tid]
        if len(history) > 0:
            px, py = history[-1]
            displacements.append(math.sqrt((cx-px)**2 + (cy-py)**2))
        history.append((cx, cy))

    stale = _active_ids - current_ids
    for sid in stale:
        _centroid_history.pop(sid, None)
    _active_ids = current_ids

    raw_v = float(np.mean(displacements)) if displacements else 0.0
    _v_history.append(raw_v)

    # ── Buffer sampling mỗi 1 giây ──────────────────────────────────
    _frame_counter += 1
    if _frame_counter >= int(ASSUMED_FPS):
        _frame_counter = 0
        _update_buffer(occupancy)

    # ── Quyết định trạng thái ────────────────────────────────────────
    label, label_vi, color = _decide_state()
    vehicle_count = len([d for d in detections if d["in_roi"]])

    # ── Vẽ lên frame ────────────────────────────────────────────────
    _draw_roi(frame, color)
    _draw_bboxes(frame, detections)
    _draw_hud(frame, occupancy, vehicle_count, label, label_vi, color)

    return frame
