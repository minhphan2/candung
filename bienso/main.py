import cv2
import easyocr
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Tạo thư mục lưu ảnh biển số cắt ra
os.makedirs("plates_crop", exist_ok=True)

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VIDEO VÀ KHỞI TẠO AI
# ==========================================
# Thay bằng đường dẫn đến file video trên máy của bạn
VIDEO_PATH = "Smart Dash Camera - Ông vua camera hành trình tại Việt Nam.mp4"

print("⏳ Đang tải mô hình YOLOv8 và EasyOCR (Có thể mất vài giây lần đầu)...")
model_yolo = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(["en"], gpu=False)
print("✅ Khởi tạo hệ thống thành công!")

# (Đã xóa hàm maximizeContrast vì không tương thích tốt với EasyOCR)

# ==========================================
# 2. BỘ LỌC VÀ SỬA LỖI BIỂN SỐ VIỆT NAM
# ==========================================
def apply_vietnamese_rules(text):
    """Sửa các lỗi nhận diện ký tự thường gặp dựa trên cấu trúc biển số VN"""
    text = text.replace(" ", "")
    
    if len(text) < 7:
        return ""
    
    # Ký tự -> Số (dùng cho mã tỉnh và phần đuôi số)
    char_to_num = {"O": "0", "D": "0", "B": "8", "Z": "2", "S": "5", "I": "1", "G": "6", "A": "4", "T": "7"}
    # Số -> Ký tự (dùng cho series chữ)
    num_to_char = {"0": "D", "8": "B", "2": "Z", "5": "S", "1": "A", "4": "A", "6": "G", "7": "T"}
    
    parts = text.split("-")
    if len(parts) == 2 and len(parts[0]) >= 2:
        prefix, suffix = list(parts[0]), list(parts[1])
        
        # 2 ký tự đầu của biển phải là SỐ (Mã tỉnh)
        for i in range(min(2, len(prefix))):
            if prefix[i] in char_to_num:
                prefix[i] = char_to_num[prefix[i]]
                
        # Ký tự thứ 3 của biển phải là CHỮ (Series)
        if len(prefix) >= 3 and prefix[2] in num_to_char:
            prefix[2] = num_to_char[prefix[2]]
            
        # Toàn bộ phần đuôi biển số (suffix) phải là SỐ
        for i in range(len(suffix)):
            if suffix[i] in char_to_num:
                suffix[i] = char_to_num[suffix[i]]
                
        return "".join(prefix) + "-" + "".join(suffix)
        
    elif len(parts) == 1:
        chars = list(text)
        # 2 ký tự đầu là SỐ
        for i in range(min(2, len(chars))):
            if chars[i] in char_to_num:
                chars[i] = char_to_num[chars[i]]
        # Ký tự thứ 3 là CHỮ
        if len(chars) >= 3 and chars[2] in num_to_char:
             chars[2] = num_to_char[chars[2]]
             
        # Các ký tự từ thứ 4 trở đi là phần ĐUÔI SỐ -> Ép thành số
        if len(chars) > 3:
            suffix = chars[3:]
            for i in range(len(suffix)):
                if suffix[i] in char_to_num:
                    suffix[i] = char_to_num[suffix[i]]
            return "".join(chars[:3]) + "-" + "".join(suffix)
            
        return text
    return text

def parse_vietnamese_plate(ocr_results):
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
        boxes.append(
            {
                "text": clean_text,
                "y_center": y_center,
                "x_min": bbox[0][0],
                "height": height,
            }
        )

    if not boxes:
        return ""

    boxes.sort(key=lambda b: b["y_center"])
    avg_height = sum(b["height"] for b in boxes) / len(boxes)
    y_diff = boxes[-1]["y_center"] - boxes[0]["y_center"]

    # Thuật toán phân tách dòng dựa vào khoảng cách trục Y
    if y_diff > 0.45 * avg_height:
        y_thresh = (boxes[0]['y_center'] + boxes[-1]['y_center']) / 2
        line1 = sorted(
            [b for b in boxes if b["y_center"] <= y_thresh],
            key=lambda b: b["x_min"],
        )
        line2 = sorted(
            [b for b in boxes if b["y_center"] > y_thresh],
            key=lambda b: b["x_min"],
        )
        raw_text = (
            "".join([b["text"] for b in line1])
            + "-"
            + "".join([b["text"] for b in line2])
        )
    else:
        boxes = sorted(boxes, key=lambda b: b["x_min"])
        raw_text = "".join([b["text"] for b in boxes])

    return apply_vietnamese_rules(raw_text)


# ==========================================
# 3. VÒNG LẶP MONITOR VIDEO THỜI GIAN THỰC
# ==========================================
def start_cctv_monitor(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Không thể mở tệp video. Vui lòng kiểm tra lại đường dẫn!")
        return

    # Kích thước cửa sổ CCTV Monitor muốn hiển thị (Full HD)
    display_w, display_h = 1280, 720

    # Tạo cửa sổ OpenCV có thể co giãn
    cv2.namedWindow("CCTV ANPR Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV ANPR Monitor", display_w, display_h)

    print("🎥 Hệ thống CCTV đang chạy... Nhấn phím 'Q' trên bàn phím để thoát.")

    # Mở file log để lưu lại kết quả
    log_file = open("bien_so_log.txt", "w", encoding="utf-8")
    log_file.write("=== KẾT QUẢ NHẬN DIỆN BIỂN SỐ ===\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🎉 Đã phát hết video clip.")
            break

        frame_idx += 1

        # [MẸO TỐI ƯU TỐC ĐỘ]: Cứ 2 frames chạy YOLO 1 lần để video hiển thị mượt mà, không bị giật hình
        if frame_idx % 2 != 0:
            # Vẫn hiển thị khung hình cũ nhưng bỏ qua bước xử lý AI nặng
            # Tỉ lệ scale tọa độ từ 4K gốc xuống màn hình 720p
            display_frame = cv2.resize(frame, (display_w, display_h))
            cv2.imshow("CCTV ANPR Monitor", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Tạo một bản sao frame để resize làm màn hình hiển thị
        display_frame = cv2.resize(frame, (display_w, display_h))
        scale_x = display_w / frame.shape[1]
        scale_y = display_h / frame.shape[0]

        # 1. Gọi YOLOv8 quét biển số trên frame 4K gốc (giữ độ nét để soi chữ từ xa)
        results = model_yolo(frame, imgsz=1280, conf=0.35, verbose=False)[0]

        # Danh sách gom TẤT CẢ biển số phát hiện trong frame này
        detected_plates = []
        list_debug_stacks = []

        for box in results.boxes:
            # Tọa độ hộp bám theo xe trên ảnh gốc 4K
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # === THU NHỎ BBOX ===
            # Thu hẹp khung cắt lại để bỏ đi phần viền đen và ốc vít ở mép biển số
            # Đặt shrink_ratio = 0.05 nghĩa là gọt đi 5% ở mỗi cạnh.
            # (Bạn bảo 0.5 có thể là quá nhiều tức là gọt đi một nửa, tôi để mặc định 0.05, bạn có thể tự chỉnh nhé!)
            shrink_ratio = 0.07
            
            bw = x2 - x1
            bh = y2 - y1
            
            x1 = int(x1 + bw * shrink_ratio)
            y1 = int(y1 + bh * shrink_ratio)
            x2 = int(x2 - bw * shrink_ratio)
            y2 = int(y2 - bh * shrink_ratio)
            # ====================

            # Ràng buộc tọa độ tránh lỗi tràn biên ảnh khi cắt
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # 2. CẮT BBOX BIỂN SỐ
            plate_crop = frame[y1_c:y2_c, x1_c:x2_c]

            # ==========================================
            # B. XỬ LÝ VÙNG CHỨA BIỂN SỐ (Bám sát Mục lục Đồ án)
            # ==========================================
            # 1. Chuyển Đổi và Làm Mờ Ảnh
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
            # 2. Tính Toán Ngưỡng (Thresholding)
            # 4. Invert (Đảo màu) kết hợp luôn ở bước này (THRESH_BINARY_INV)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 3. Morphological Transformations
            # Dùng phép đóng (Closing) để nối liền các đứt gãy của nét chữ
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Đảo ngược lại lần nữa để chữ Đen nền Trắng cho EasyOCR dễ đọc
            morph = cv2.bitwise_not(morph)

            # === DEBUG VISUALIZATION: XEM TRỰC QUAN TỪNG BƯỚC ===
            # Chuyển tất cả về chuẩn 3 kênh màu (BGR) để ghép lại với nhau
            debug_gray   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            debug_blur   = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
            debug_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            debug_morph  = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            
            # Viết chữ lên từng ảnh để dễ phân biệt
            cv2.putText(debug_gray, "1. GRAY (Resize x3)", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_blur, "2. BLUR", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_thresh, "3. OTSU THRESH", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_morph, "4. MORPH (Final)", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Ghép ngang 4 ảnh lại thành 1 dải ảnh dài
            import numpy as np
            debug_stack = np.hstack((debug_gray, debug_blur, debug_thresh, debug_morph))
            
            # Resize để ghép dọc nếu có nhiều xe
            target_w = 800
            h, w = debug_stack.shape[:2]
            target_h = int((target_w / w) * h)
            debug_stack_resized = cv2.resize(debug_stack, (target_w, target_h))
            list_debug_stacks.append(debug_stack_resized)
            # ====================================================

            # 4. Nhận dạng chữ (OCR)
            # Ở đây chúng ta dùng EasyOCR thay cho Tesseract để không phải cài đặt phức tạp mà độ chính xác cao hơn!
            allowlist = '0123456789ABCDEFGHJKLMNPRSTUVXYZ-'
            ocr_res = reader.readtext(morph, allowlist=allowlist, paragraph=False)
            plate_text = parse_vietnamese_plate(ocr_res)
            
            # Đưa qua bộ luật biển số VN để chuẩn hóa
            if plate_text:
                plate_text = apply_vietnamese_rules(plate_text)

            # Chuyển đổi tọa độ để vẽ lên màn hình
            rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
            rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)

            if plate_text:
                detected_plates.append(plate_text)

                # Vẽ BBox màu XANH LÁ nếu đọc được chữ
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

                # Dải nền và chữ
                label_size, base_line = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                top_y = max(ry1, label_size[1] + 10)
                cv2.rectangle(display_frame, (rx1, top_y - label_size[1] - 8),
                              (rx1 + label_size[0] + 6, top_y + base_line), (0, 255, 0), cv2.FILLED)
                cv2.putText(display_frame, plate_text, (rx1 + 3, top_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                # Vẽ BBox màu ĐỎ nếu YOLO bắt được nhưng OCR KHÔNG ĐỌC ĐƯỢC CHỮ
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

        # SAU KHI QUÉT XONG TẤT CẢ CÁC XE TRONG FRAME -> IN 1 DÒNG DUY NHẤT
        if detected_plates:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            all_plates = " | ".join(detected_plates)
            print(f"[CCTV Frame {frame_idx}] {all_plates}")
            log_file.write(f"[{now_str}] Frame {frame_idx}: {all_plates}\n")
            log_file.flush()

        # Hiển thị cửa sổ Debug nếu có biển số
        if list_debug_stacks:
            final_debug_window = np.vstack(list_debug_stacks)
            cv2.imshow("Debug: Processing Pipeline", final_debug_window)

        # Đẩy khung hình đã vẽ thông tin lên cửa sổ GUI thời gian thực
        cv2.imshow("CCTV ANPR Monitor", display_frame)

        # Nhấn phím 'q' để dừng clip lập tức. Nếu phát hiện biển số, tự làm chậm video lại (delay 500ms)
        delay_time = 500 if detected_plates else 1
        if cv2.waitKey(delay_time) & 0xFF == ord("q"):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


# Kích hoạt chương trình
if __name__ == "__main__":
    start_cctv_monitor(VIDEO_PATH)