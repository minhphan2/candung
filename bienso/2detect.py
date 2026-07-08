import cv2
from ultralytics import YOLO

# Khởi tạo 2 mô hình YOLO
print("⏳ Đang tải mô hình YOLO Tầng 1 (Bắt biển số)...")
plate_model = YOLO('license_plate_detector.pt')

print("⏳ Đang tải mô hình YOLO Tầng 2 (Bắt ký tự)...")
char_model = YOLO('best_char_cnn_softmax.pth')

VIDEO_PATH = "Smart Dash Camera - Ông vua camera hành trình tại Việt Nam.mp4"

def process_plate(plate_crop):
    """
    Nhận vào ảnh cắt của biển số, dùng char_model để tìm các ký tự
    và sắp xếp chúng thành chuỗi biển số hoàn chỉnh.
    """
    # Chạy YOLO để tìm ký tự trên ảnh biển số
    # Kích thước ảnh biển số nhỏ, nên dùng imgsz nhỏ (320) để tăng tốc
    results = char_model(plate_crop, imgsz=320, conf=0.25, verbose=False)[0]
    
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        # Lấy tên của class (vd: 'A', '1', '2'...).
        # CHÚ Ý: Nếu file yolo11n.pt chưa được train, nó sẽ ra tên 'person', 'car'...
        char_name = char_model.names[cls_id] 
        
        y_center = (y1 + y2) / 2
        height = y2 - y1
        boxes.append({
            "char": char_name,
            "x_min": x1,
            "y_center": y_center,
            "height": height
        })
        
    if not boxes:
        return ""
        
    # Sắp xếp logic 1 dòng / 2 dòng
    boxes.sort(key=lambda b: b["y_center"])
    avg_height = sum(b["height"] for b in boxes) / len(boxes)
    y_diff = boxes[-1]["y_center"] - boxes[0]["y_center"]
    
    # Nếu khoảng cách Y giữa chữ cao nhất và thấp nhất lớn hơn 45% chiều cao trung bình -> Biển 2 dòng
    if y_diff > 0.45 * avg_height:
        y_thresh = (boxes[0]['y_center'] + boxes[-1]['y_center']) / 2
        line1 = sorted([b for b in boxes if b["y_center"] <= y_thresh], key=lambda b: b["x_min"])
        line2 = sorted([b for b in boxes if b["y_center"] > y_thresh], key=lambda b: b["x_min"])
        plate_text = "".join([b["char"] for b in line1]) + "-" + "".join([b["char"] for b in line2])
    else:
        # Biển 1 dòng
        boxes = sorted(boxes, key=lambda b: b["x_min"])
        plate_text = "".join([b["char"] for b in boxes])
        
    return plate_text

def run_2_stage_yolo():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở video.")
        return
        
    cv2.namedWindow("2-Stage YOLO Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2-Stage YOLO Monitor", 1280, 720)
    
    print("✅ Hệ thống chạy 2 tầng YOLO đang khởi động... Nhấn 'q' để thoát.")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Nhảy frame để chống giật (bỏ qua frame lẻ)
        if frame_idx % 2 != 0:
            continue
            
        display_frame = frame.copy()
            
        # 1. Phát hiện biển số (Tầng 1)
        plate_results = plate_model(frame, imgsz=1280, conf=0.35, verbose=False)[0]
        
        for box in plate_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Ràng buộc tọa độ
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            plate_crop = frame[y1_c:y2_c, x1_c:x2_c]
            if plate_crop.size == 0:
                continue
                
            # 2. Quét ký tự trên biển số (Tầng 2)
            plate_text = process_plate(plate_crop)
            
            if plate_text:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label_size, base_line = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                top_y = max(y1, label_size[1] + 10)
                
                cv2.rectangle(display_frame, (x1, top_y - label_size[1] - 10), 
                              (x1 + label_size[0] + 10, top_y + base_line), (0, 255, 0), cv2.FILLED)
                cv2.putText(display_frame, plate_text, (x1 + 5, top_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
        # Resize lại khung hình trước khi show để vừa màn hình
        display_frame_resized = cv2.resize(display_frame, (1280, 720))
        cv2.imshow("2-Stage YOLO Monitor", display_frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_2_stage_yolo()
