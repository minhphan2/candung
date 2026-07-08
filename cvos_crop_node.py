"""
cvos_crop_node.py — CVOS Python Function Node: Crop License Plate

Node này là mắt xích TRUNG GIAN trong pipeline:
    [YOLO ONNX Node] --Bbox--> [crop_plate_node] --Image--> [lpr_ocr_node]

Nhận vào:
    - image : Image  → Frame đầy đủ từ camera
    - bbox  : Bbox   → Tọa độ các biển số phát hiện được (output từ YOLO ONNX Node)

Trả ra:
    - Image → Ảnh biển số đầu tiên đã được cắt ra, sẵn sàng cho OCR Node.

Cách upload lên CVOS:
    1. Dashboard → Python Import → chọn file này
    2. CVOS tự detect hàm `crop_plate_node`
    3. Input Topic 1 (image): camera_frame  (nối với Camera Source Node)
    4. Input Topic 2 (bbox) : plate_bbox    (nối với YOLO ONNX Node output)
    5. Output Topic         : plate_crop    (nối với lpr_ocr_node input)
"""

import cv2
import numpy as np
from cvos.topic.types_lib import Image, Bbox


def crop_plate_node(image: Image, bbox: Bbox) -> Image:
    """
    Cắt vùng biển số từ frame đầy đủ dựa trên tọa độ BBox của YOLO.

    CVOS sẽ tự động đồng bộ (sync) hai Topic đầu vào (image + bbox)
    theo seq_no trước khi gọi hàm này — đảm bảo frame và bbox luôn khớp nhau.

    Chiến lược: Lấy biển số có confidence CAO NHẤT trong frame để xử lý.
    Nếu không phát hiện biển số nào → trả về ảnh đen (frame trống).
    """
    frame = image.copy()

    # bbox.boxes  : shape (N, 4) — [x1, y1, x2, y2] (tọa độ pixel)
    # bbox.scores : shape (N,)   — confidence score [0.0 – 1.0]
    # bbox.class_ids: shape (N,) — class id (thường = 0 cho biển số)
    boxes  = bbox.boxes    # numpy array
    scores = bbox.scores   # numpy array

    # Nếu YOLO không tìm thấy biển số nào trong frame này
    if boxes is None or len(boxes) == 0:
        # Trả về ảnh đen để OCR Node bỏ qua
        return np.zeros_like(frame)

    # Lấy biển số có confidence CAO NHẤT
    best_idx = int(np.argmax(scores))
    x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]

    # Ràng buộc tọa độ tránh lỗi tràn biên ảnh
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Kiểm tra vùng crop có hợp lệ không (tránh crop ảnh kích thước 0)
    if x2 <= x1 or y2 <= y1:
        return np.zeros_like(frame)

    # Cắt vùng biển số
    plate_crop = frame[y1:y2, x1:x2]
    return plate_crop
