"""
main.py — Entry Point for Real-Time Traffic Congestion Detection.

Usage:
    python main.py                          # Use defaults from Config
    python main.py --source traffic.mp4     # Specify a video file
    python main.py --source 0              # Use webcam
    python main.py --model yolov8n.pt      # Specify model weights

Press 'q' to quit, 'p' to pause/resume.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from config import Config
from pipeline import CongestionPipeline
from visualization import Visualization


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults from Config."""
    parser = argparse.ArgumentParser(
        description="Real-Time Traffic Congestion Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file path or camera index (0, 1, ...). Default: Config.VIDEO_SOURCE",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model weights (.pt). Default: Config.MODEL_PATH",
    )
    parser.add_argument(
        "--occ-thresh",
        type=float,
        default=None,
        help="Occupancy high threshold (0.0–1.0). Default: 0.55",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Number of 1-second slots in the congestion buffer. Default: 20",
    )
    parser.add_argument(
        "--cm-thresh",
        type=float,
        default=None,
        help="Current Mean threshold to confirm gridlock (0.0–1.0). Default: 0.6",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help=(
            "ROI polygon as comma-separated coords: "
            "'x1,y1,x2,y2,x3,y3,...'. Must have ≥3 vertices."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no cv2.imshow). Useful for benchmarking.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """
    Construct a Config from CLI args, falling back to defaults.
    """
    cfg = Config()

    if args.source is not None:
        # Interpret numeric strings as camera indices
        try:
            cfg.VIDEO_SOURCE = int(args.source)
        except ValueError:
            cfg.VIDEO_SOURCE = args.source

    if args.model is not None:
        cfg.MODEL_PATH = args.model

    if args.occ_thresh is not None:
        cfg.OCCUPANCY_HIGH = args.occ_thresh

    if args.buffer_size is not None:
        cfg.BUFFER_SIZE = args.buffer_size

    if args.cm_thresh is not None:
        cfg.CONGESTION_MEAN_THRESHOLD = args.cm_thresh

    if args.roi is not None:
        coords = list(map(int, args.roi.split(",")))
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError("--roi must have ≥3 vertices as 'x1,y1,x2,y2,x3,y3,...'")
        cfg.ROI_POLYGON = np.array(coords, dtype=np.int32).reshape(-1, 2)

    return cfg


def main() -> None:
    """
    Main loop:
        1. Open video source.
        2. For each frame → pipeline.process_frame() → visualization.draw().
        3. Display annotated frame; handle keyboard input.
    """
    args = parse_args()
    cfg = build_config(args)

    # ── Initialize components ────────────────────────────────────────
    print("=" * 60)
    print("  TRAFFIC CONGESTION DETECTION PIPELINE")
    print("=" * 60)
    print(f"  Model      : {cfg.MODEL_PATH}")
    print(f"  Source      : {cfg.VIDEO_SOURCE}")
    print(f"  ROI verts   : {len(cfg.ROI_POLYGON)} points")
    print(f"  Occ thresh  : {cfg.OCCUPANCY_HIGH}")
    print(f"  Buffer      : {cfg.BUFFER_SIZE} slots x 1s")
    print(f"  CM thresh   : {cfg.CONGESTION_MEAN_THRESHOLD}")
    print(f"  Tracker     : {cfg.TRACKER_TYPE}")
    print("=" * 60)

    # ── Open video capture ───────────────────────────────────────
    cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {cfg.VIDEO_SOURCE}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pipeline = CongestionPipeline(cfg, fps=fps_video)
    viz = Visualization(cfg)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {width}x{height} @ {fps_video:.1f} FPS, {total_frames} frames")
    print("-" * 60)

    paused = False
    frame_idx = 0

    try:
        while True:
            # ── Handle pause ─────────────────────────────────────────
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("p"):
                    paused = False
                elif key == ord("q"):
                    break
                continue

            # ── Read frame ───────────────────────────────────────────
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            # ── Optional resize ──────────────────────────────────────
            if cfg.DISPLAY_WIDTH is not None:
                scale = cfg.DISPLAY_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # ── Run pipeline ─────────────────────────────────────────
            frame, metrics, detections = pipeline.process_frame(frame)

            # ── Draw overlays ────────────────────────────────────────
            annotated = viz.draw(frame, metrics, detections)

            # ── FPS counter (bottom-right) ───────────────────────────
            elapsed = time.perf_counter() - t_start
            fps_actual = 1.0 / max(elapsed, 1e-6)
            fps_text = f"FPS: {fps_actual:.1f}"
            h, w = annotated.shape[:2]
            cv2.putText(
                annotated, fps_text,
                (w - 160, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 255), 2, cv2.LINE_AA,
            )

            # ── Time counter (seconds) ────────────────────────────────
            frame_idx += 1
            elapsed_sec = frame_idx / fps_video
            time_info = f"Time: {elapsed_sec:.1f}s"
            if total_frames > 0:
                total_sec = total_frames / fps_video
                time_info += f" / {total_sec:.1f}s"
            cv2.putText(
                annotated, time_info,
                (w - 250, h - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA,
            )

            # ── Console log (every ~1 second) ─────────────────────
            log_interval = max(int(fps_video), 1)  # log once per second
            if frame_idx % log_interval == 0:
                buf_str = ''.join(str(b) for b in pipeline._buffer)
                print(
                    f"  [{elapsed_sec:>7.1f}s] "
                    f"O={metrics.occupancy:.3f}  "
                    f"CM={metrics.buffer_cm:.2f}  "
                    f"Buf=[{buf_str}]  "
                    f"State={metrics.state.label:<12s}  "
                    f"x{metrics.vehicle_count}"
                )

            # ── Display ──────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow("Traffic Congestion Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    paused = True

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  Processed {frame_idx} frames. Exiting.")


if __name__ == "__main__":
    main()
