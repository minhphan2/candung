"""
visualization.py — Overlay Rendering & HUD for Congestion Pipeline.

Responsible for all drawing: ROI polygon, tracked bounding boxes,
and a professional dashboard panel showing real-time metrics.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import Config
from pipeline import Metrics, TrafficState


class Visualization:
    """
    Draws overlays on each frame:
        1. ROI polygon — color-coded by current congestion state.
        2. Tracked vehicle bounding boxes with ID labels.
        3. Top-left HUD dashboard: Occupancy %, Motion Score, State.
    """

    # ── HUD Layout Constants ─────────────────────────────────────────
    HUD_X = 16                # left margin
    HUD_Y = 16                # top margin
    HUD_W = 380               # panel width
    HUD_H = 248               # panel height
    HUD_BG_ALPHA = 0.72       # background transparency
    HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
    HUD_FONT_SCALE_TITLE = 0.70
    HUD_FONT_SCALE_BODY = 0.60
    HUD_FONT_THICKNESS = 2
    HUD_LINE_SPACING = 32     # vertical gap between text lines

    # ── BBox Colors per class ────────────────────────────────────────
    BBOX_COLORS: dict[int, tuple[int, int, int]] = {
        2: (255, 180, 50),    # Car — blue-ish
        3: (50, 255, 200),    # Motorcycle — cyan
        5: (255, 100, 255),   # Bus — magenta
        7: (100, 200, 255),   # Truck — orange
    }
    DEFAULT_BBOX_COLOR = (200, 200, 200)

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────
    def draw(
        self,
        frame: np.ndarray,
        metrics: Metrics,
        detections: list[dict],
    ) -> np.ndarray:
        """
        Compose all visual overlays onto the frame (in-place).

        Args:
            frame: BGR image to annotate.
            metrics: Current-frame metrics snapshot.
            detections: List of detection dicts from the pipeline.

        Returns:
            The annotated frame (same object, modified in-place).
        """
        self._draw_roi(frame, metrics.state)
        self._draw_bboxes(frame, detections)
        self._draw_hud(frame, metrics)
        return frame

    # ─────────────────────────────────────────────────────────────────
    # ROI Polygon Overlay
    # ─────────────────────────────────────────────────────────────────
    def _draw_roi(self, frame: np.ndarray, state: TrafficState) -> None:
        """
        Draw the ROI polygon as a semi-transparent filled region
        with a solid border. Color is determined by the current traffic state.
        """
        overlay = frame.copy()
        color = state.color_bgr
        pts = self.cfg.ROI_POLYGON.reshape((-1, 1, 2))

        # Filled polygon (semi-transparent)
        cv2.fillPoly(overlay, [self.cfg.ROI_POLYGON], color)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # Solid border
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)

    # ─────────────────────────────────────────────────────────────────
    # Tracked Bounding Boxes
    # ─────────────────────────────────────────────────────────────────
    def _draw_bboxes(self, frame: np.ndarray, detections: list[dict]) -> None:
        """
        Draw bounding boxes and tracking IDs for all detected vehicles.
        Vehicles inside the ROI get a thicker outline + filled label.
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            tid = det["track_id"]
            cls_id = det["cls_id"]
            conf = det["conf"]
            in_roi = det.get("in_roi", False)

            color = self.BBOX_COLORS.get(cls_id, self.DEFAULT_BBOX_COLOR)
            thickness = 3 if in_roi else 1

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label: "ID:5 Car 0.87"
            cls_name = self.cfg.CLASS_NAMES.get(cls_id, "?")
            label = f"ID:{tid} {cls_name} {conf:.2f}"

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, self.HUD_FONT, 0.45, 1
            )
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 6, label_y + baseline),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 3, label_y - 2),
                self.HUD_FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

    # ─────────────────────────────────────────────────────────────────
    # Dashboard / HUD Panel
    # ─────────────────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, metrics: Metrics) -> None:
        """
        Render a professional semi-transparent dashboard panel
        in the top-left corner showing:
            • Congestion State (with color indicator)
            • Occupancy Ratio (%)
            • Motion Score (V px/frame)
            • Vehicle Count in ROI
        """
        x, y = self.HUD_X, self.HUD_Y
        w, h = self.HUD_W, self.HUD_H

        # ── Semi-transparent dark background ─────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.HUD_BG_ALPHA, frame, 1 - self.HUD_BG_ALPHA, 0, frame)

        # Thin border
        cv2.rectangle(frame, (x, y), (x + w, y + h), metrics.state.color_bgr, 2)

        # ── Title bar ────────────────────────────────────────────────
        ty = y + 28
        cv2.putText(
            frame,
            "TRAFFIC MONITOR",
            (x + 10, ty),
            self.HUD_FONT, self.HUD_FONT_SCALE_TITLE,
            (255, 255, 255), self.HUD_FONT_THICKNESS, cv2.LINE_AA,
        )

        # Separator line
        ty += 10
        cv2.line(frame, (x + 10, ty), (x + w - 10, ty), (100, 100, 100), 1)

        # ── State indicator ──────────────────────────────────────────
        ty += self.HUD_LINE_SPACING
        state_text = f"State: {metrics.state.label}"
        cv2.putText(
            frame, state_text,
            (x + 10, ty),
            self.HUD_FONT, self.HUD_FONT_SCALE_BODY,
            metrics.state.color_bgr, self.HUD_FONT_THICKNESS, cv2.LINE_AA,
        )

        # Vietnamese sub-label
        ty += 22
        cv2.putText(
            frame, f"       ({metrics.state.label_vi})",
            (x + 10, ty),
            self.HUD_FONT, 0.45,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # ── Occupancy bar ────────────────────────────────────────────
        ty += self.HUD_LINE_SPACING
        occ_pct = metrics.occupancy * 100
        occ_text = f"Occupancy: {occ_pct:5.1f}%"
        cv2.putText(
            frame, occ_text,
            (x + 10, ty),
            self.HUD_FONT, self.HUD_FONT_SCALE_BODY,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Mini progress bar for occupancy
        bar_x = x + 220
        bar_w = 140
        bar_h = 14
        bar_y = ty - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * min(metrics.occupancy, 1.0))
        bar_color = metrics.state.color_bgr
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        # ── Buffer Current Mean (CM) ───────────────────────────────────
        ty += self.HUD_LINE_SPACING
        cm = metrics.buffer_cm
        cm_thresh = self.cfg.CONGESTION_MEAN_THRESHOLD
        cm_text = f"CM: {cm:.2f} / {cm_thresh:.2f}"
        cv2.putText(
            frame, cm_text,
            (x + 10, ty),
            self.HUD_FONT, self.HUD_FONT_SCALE_BODY,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Mini progress bar for CM
        bar_x = x + 220
        bar_w = 140
        bar_h = 14
        bar_y = ty - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * min(cm, 1.0))
        # Màu: xanh lá khi thấp → vàng → đỏ khi >= ngưỡng
        if cm >= cm_thresh:
            cm_color = (0, 0, 200)       # Đỏ
        elif cm >= cm_thresh * 0.5:
            cm_color = (0, 200, 255)     # Vàng
        else:
            cm_color = (0, 200, 0)       # Xanh lá
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), cm_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

        # ── Vehicle count ────────────────────────────────────────────
        ty += self.HUD_LINE_SPACING
        count_text = f"Vehicles in ROI: {metrics.vehicle_count}"
        cv2.putText(
            frame, count_text,
            (x + 10, ty),
            self.HUD_FONT, self.HUD_FONT_SCALE_BODY,
            (0, 220, 255), self.HUD_FONT_THICKNESS, cv2.LINE_AA,
        )
