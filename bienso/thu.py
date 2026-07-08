"""
thu.py — Pipeline nhận diện biển số xe Việt Nam hoàn chỉnh

Luồng xử lý:
    1. license_plate_detector.pt  → Phát hiện vùng biển số trên ảnh/video
    2. Cắt (Crop) vùng biển số ra
    3. best_char.pt               → Phát hiện từng ký tự trên biển số đã cắt
    4. K-Means phân 2 dòng        → Ghép thành chuỗi biển số chuẩn VN

Cách chạy:
    python thu.py                          # Dùng webcam (mặc định)
    python thu.py --source anh.jpg         # Dùng ảnh
    python thu.py --source video.mp4       # Dùng video
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════
PLATE_MODEL_PATH = "license_plate_detector.pt"   # Model phát hiện biển số
CHAR_MODEL_PATH  = "best_char2.pt"                # Model nhận diện ký tự
PLATE_CONF       = 0.40                          # Ngưỡng confidence biển số
CHAR_CONF        = 0.45                          # Ngưỡng confidence ký tự
PADDING          = 0                            # Pixel mở rộng vùng crop


# ═══════════════════════════════════════════════════════════════════════
# NẠP MODEL
# ═══════════════════════════════════════════════════════════════════════
plate_model = YOLO(PLATE_MODEL_PATH)
char_model  = YOLO(CHAR_MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════
# HÀM XỬ LÝ
# ═══════════════════════════════════════════════════════════════════════

def crop_plate(frame, box):
    """Cắt vùng biển số từ frame, có padding để không bị sát viền."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(w, x2 + PADDING)
    y2 = min(h, y2 + PADDING)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def read_plate(plate_img):
    """
    Nhận ảnh biển số đã cắt → Chạy best_char.pt → Phân 2 dòng bằng K-Means → Trả chuỗi biển số.
    
    Trả về:
        plate_text : str  — Chuỗi biển số (ví dụ: "29A1-12345")
        char_boxes : list — Danh sách ký tự kèm tọa độ (để vẽ lên ảnh nếu cần)
    """
    results = char_model.predict(source=plate_img, conf=CHAR_CONF, verbose=False)

    detected_chars = []
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = [int(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            conf     = float(box.conf[0])
            char_label = char_model.names[class_id]
            y_center = (ymin + ymax) / 2.0
            x_center = (xmin + xmax) / 2.0

            detected_chars.append({
                "char": char_label,
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax,
                "x_center": x_center,
                "y_center": y_center,
                "conf": conf,
            })

    if len(detected_chars) == 0:
        return "", []

    # ── Phân 2 dòng bằng K-Means ────────────────────────────────────
    if len(detected_chars) >= 4:
        y_centers = np.array([c["y_center"] for c in detected_chars], dtype=np.float32).reshape(-1, 1)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(y_centers, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Dòng trên = nhóm có center Y nhỏ hơn
        top_idx = 0 if centers[0][0] < centers[1][0] else 1

        line1 = [c for i, c in enumerate(detected_chars) if labels[i][0] == top_idx]
        line2 = [c for i, c in enumerate(detected_chars) if labels[i][0] != top_idx]

        # Sắp xếp trái → phải
        line1.sort(key=lambda c: c["xmin"])
        line2.sort(key=lambda c: c["xmin"])

        text1 = "".join([c["char"] for c in line1]).upper()
        text2 = "".join([c["char"] for c in line2]).upper()
        plate_text = f"{text1}-{text2}"
    else:
        # Ít ký tự quá (biển 1 dòng hoặc lỗi) → sắp xếp trái qua phải
        detected_chars.sort(key=lambda c: c["xmin"])
        plate_text = "".join([c["char"] for c in detected_chars]).upper()

    return plate_text, detected_chars


def draw_results(frame, plate_bbox, plate_text, char_boxes, plate_img_shape):
    """Vẽ BBox biển số + text lên frame gốc."""
    x1, y1, x2, y2 = plate_bbox

    # Vẽ khung biển số
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Vẽ nền cho text
    label = plate_text if plate_text else "???"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Vẽ từng ký tự lên vùng biển số (tùy chọn, để debug)
    for c in char_boxes:
        cx1 = x1 + c["xmin"]
        cy1 = y1 + c["ymin"]
        cx2 = x1 + c["xmax"]
        cy2 = y1 + c["ymax"]
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 1)
        cv2.putText(frame, c["char"], (cx1, cy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════
# VÒNG LẶP CHÍNH
# ═══════════════════════════════════════════════════════════════════════

def process_frame(frame):
    """Xử lý 1 frame: Detect biển số → Crop → OCR → Vẽ kết quả."""
    results = plate_model.predict(source=frame, conf=PLATE_CONF, verbose=False)

    for result in results:
        for box in result.boxes:
            # Bước 1: Cắt vùng biển số
            plate_img, plate_bbox = crop_plate(frame, box)

            if plate_img.size == 0:
                continue

            # Bước 2: Đọc ký tự trên biển số
            plate_text, char_boxes = read_plate(plate_img)

            # Bước 3: Vẽ kết quả lên frame gốc
            frame = draw_results(frame, plate_bbox, plate_text, char_boxes, plate_img.shape)

            # In ra terminal
            if plate_text:
                plate_conf = float(box.conf[0])
                print(f"[BIỂN SỐ] {plate_text}  (conf: {plate_conf:.2f})")

    return frame


def run(source):
    """Chạy pipeline trên webcam / video / ảnh."""

    # Nếu source là số → webcam
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # Nếu là ảnh tĩnh
    if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(source)
        if frame is None:
            print(f"❌ Không đọc được ảnh: {source}")
            return
        result = process_frame(frame)
        cv2.imshow("Thu - Bien So", result)
        print("\nNhấn phím bất kỳ để thoát...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video hoặc webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Không mở được nguồn video: {source}")
        return

    print(f"🎥 Đang chạy nhận diện biển số... (Nhấn 'q' để thoát)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)
        cv2.imshow("Thu - Bien So", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Đã dừng.")


# ═══════════════════════════════════════════════════════════════════════
# ĐIỂM VÀO
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline nhận diện biển số xe Việt Nam")
    parser.add_argument("--source", default="0", help="Đường dẫn video/ảnh hoặc 0 cho webcam (mặc định: 0)")
    args = parser.parse_args()

    run(args.source)
