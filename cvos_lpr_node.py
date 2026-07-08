"""
File này là CVOS Python Function Node, dùng để import vào CVOS Dashboard.

Cách upload:
1. Vào Dashboard > Python Import
2. Chọn file này
3. CVOS sẽ tự detect hàm `lpr_ocr_node`
4. Điền Input Topic = topic camera của bạn (ví dụ: camera_frame)
5. Điền Output Topic = tên output bạn muốn (ví dụ: annotated_frame)
"""

import cv2
import easyocr
import numpy as np
from cvos.topic.types_lib import Image

# ==========================================
# KHỞI TẠO MODEL (Lazy - chỉ load 1 lần duy nhất)
# ==========================================
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        # gpu=True vì đây là Jetson (có GPU)
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


# ==========================================
# BỘ LỌC LUẬT BIỂN SỐ VIỆT NAM
# ==========================================
def _apply_vietnamese_rules(text):
    """Sửa các lỗi nhận diện ký tự thường gặp dựa trên cấu trúc biển số VN"""
    text = text.replace(" ", "")
    if len(text) < 7:
        return ""

    char_to_num = {"O": "0", "D": "0", "B": "8", "Z": "2", "S": "5", "I": "1", "G": "6", "A": "4", "T": "7"}
    num_to_char = {"0": "D", "8": "B", "2": "Z", "5": "S", "1": "A", "4": "A", "6": "G", "7": "T"}

    parts = text.split("-")
    if len(parts) == 2 and len(parts[0]) >= 2:
        prefix, suffix = list(parts[0]), list(parts[1])
        for i in range(min(2, len(prefix))):
            if prefix[i] in char_to_num:
                prefix[i] = char_to_num[prefix[i]]
        if len(prefix) >= 3 and prefix[2] in num_to_char:
            prefix[2] = num_to_char[prefix[2]]
        for i in range(len(suffix)):
            if suffix[i] in char_to_num:
                suffix[i] = char_to_num[suffix[i]]
        return "".join(prefix) + "-" + "".join(suffix)

    elif len(parts) == 1:
        chars = list(text)
        for i in range(min(2, len(chars))):
            if chars[i] in char_to_num:
                chars[i] = char_to_num[chars[i]]
        if len(chars) >= 3 and chars[2] in num_to_char:
            chars[2] = num_to_char[chars[2]]
        if len(chars) > 3:
            suffix = chars[3:]
            for i in range(len(suffix)):
                if suffix[i] in char_to_num:
                    suffix[i] = char_to_num[suffix[i]]
            return "".join(chars[:3]) + "-" + "".join(suffix)
        return text
    return text


def _parse_plate(ocr_results):
    """Phân loại và sắp xếp chữ cho biển 1 dòng và biển 2 dòng"""
    if not ocr_results:
        return ""
    boxes = []
    for bbox, text, prob in ocr_results:
        clean_text = "".join(e for e in text if e.isalnum()).upper()
        if not clean_text or prob < 0.25:
            continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        height = bbox[2][1] - bbox[0][1]
        boxes.append({"text": clean_text, "y_center": y_center, "x_min": bbox[0][0], "height": height})

    if not boxes:
        return ""

    boxes.sort(key=lambda b: b["y_center"])
    avg_height = sum(b["height"] for b in boxes) / len(boxes)
    y_diff = boxes[-1]["y_center"] - boxes[0]["y_center"]

    if y_diff > 0.45 * avg_height:
        y_thresh = (boxes[0]["y_center"] + boxes[-1]["y_center"]) / 2
        line1 = sorted([b for b in boxes if b["y_center"] <= y_thresh], key=lambda b: b["x_min"])
        line2 = sorted([b for b in boxes if b["y_center"] > y_thresh], key=lambda b: b["x_min"])
        raw_text = "".join([b["text"] for b in line1]) + "-" + "".join([b["text"] for b in line2])
    else:
        boxes = sorted(boxes, key=lambda b: b["x_min"])
        raw_text = "".join([b["text"] for b in boxes])

    return _apply_vietnamese_rules(raw_text)


# ==========================================
# CVOS NODE FUNCTION
# Đây là hàm CVOS sẽ tự động detect và wrap thành 1 Node độc lập.
# Nhận 1 Image từ Topic Camera -> Xử lý -> Trả ra 1 Image đã được vẽ kết quả.
# ==========================================
def lpr_ocr_node(plate_crop: Image) -> Image:
    """
    Nhận ảnh biển số đã được cắt ra (từ Node YOLO ONNX trước đó),
    tiền xử lý và chạy EasyOCR để đọc ký tự.
    Trả về ảnh biển số đã vẽ đè kết quả nhận diện lên.
    """
    reader = _get_reader()
    frame = plate_crop.copy()

    # B.1 Chuyển Đổi và Làm Mờ Ảnh
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # B.2 Tính Toán Ngưỡng (Thresholding) + B.4 Invert
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # B.3 Morphological Transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.bitwise_not(morph)

    # B.4 Nhận dạng chữ bằng EasyOCR (thay Tesseract)
    allowlist = "0123456789ABCDEFGHJKLMNPRSTUVXYZ-"
    ocr_res = reader.readtext(morph, allowlist=allowlist, paragraph=False)
    plate_text = _parse_plate(ocr_res)

    # Vẽ kết quả lên ảnh đầu vào để trả về
    if plate_text:
        label_size, base_line = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (0, 0), (label_size[0] + 6, label_size[1] + 12), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, plate_text, (3, label_size[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return frame
