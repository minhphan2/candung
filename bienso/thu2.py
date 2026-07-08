import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Cấu hình YOLO tìm biển số và EasyOCR đọc chữ
print("⏳ Đang tải mô hình YOLOv8 và EasyOCR...")
model_yolo = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(["en"], gpu=False)
print("✅ Khởi tạo xong!")

# Các tham số chuẩn từ repo VIETNAMESE_LICENSE_PLATE
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

def maximizeContrast(imgGrayscale):
    # Bước 3: Tăng độ tương phản bằng TopHat và BlackHat
    height, width = imgGrayscale.shape
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10)
    
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat

def segment_characters(plate_crop):
    """
    Hàm này áp dụng đúng sơ đồ của Github repo lên vùng biển số đã được YOLO cắt ra
    để tìm từng chữ cái bằng Contour (không dùng AI)
    """
    height, width, _ = plate_crop.shape
    roiarea = height * width

    # Phóng to ảnh để xử lý mịn hơn
    plate_crop = cv2.resize(plate_crop, (0, 0), fx=3, fy=3)
    height, width, _ = plate_crop.shape
    roiarea = height * width

    # 1. Chuyển sang ảnh xám (Hệ HSV)
    imgHSV = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    _, _, imgGrayscale = cv2.split(imgHSV)
    
    # 2. Tăng độ tương phản
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    
    # 3. Giảm nhiễu bằng Gauss
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    
    # 4. Nhị phân hóa ảnh với ngưỡng động
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    
    # 5. Dilation để nối nét chữ (Theo code Github Image_test2.py dòng 119)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
    
    # 6. Tìm viền Contour để tách chữ
    cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_chars = []
    
    # Lọc Contour để lấy chữ (Bỏ qua ốc vít, viền)
    Min_char = 0.01  # Chữ chiếm ít nhất 1% diện tích biển
    Max_char = 0.09  # Chữ chiếm nhiều nhất 9% diện tích biển
    
    for c in cont:
        (x, y, w, h) = cv2.boundingRect(c)
        ratiochar = w / h
        char_area = w * h
        
        # Điều kiện hình học của 1 chữ cái chuẩn
        if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
            detected_chars.append((x, y, w, h))
            # Vẽ BBox chữ cái (màu đỏ)
            cv2.rectangle(plate_crop, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return plate_crop, thre_mor

# ========================================================
# CHẠY THỬ TRÊN VIDEO
# ========================================================
if __name__ == "__main__":
    video_path = "Smart Dash Camera - Ông vua camera hành trình tại Việt Nam.mp4"
    cap = cv2.VideoCapture(video_path)

    print("🎥 Bắt đầu quét video: YOLO (Tìm biển) + OpenCV Contour (Tách chữ)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (1024, 576))
        
        # BƯỚC 1: Dùng YOLO tìm Biển số (Cắt BBox)
        results = model_yolo(frame, imgsz=640, conf=0.4, verbose=False)[0]
        
        # Danh sách chứa ảnh các biển số trong 1 frame
        list_plate_results = []
        list_ocr_imgs = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # === THU NHỎ BBOX (Shrink) ===
            shrink_ratio = 0.07
            bw = x2 - x1
            bh = y2 - y1
            
            x1 = int(x1 + bw * shrink_ratio)
            y1 = int(y1 + bh * shrink_ratio)
            x2 = int(x2 - bw * shrink_ratio)
            y2 = int(y2 - bh * shrink_ratio)
            # =============================
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue

            # BƯỚC 2: Dùng thuật toán Github (Contour) để tách chữ cái
            plate_result, plate_thresh = segment_characters(plate_crop)
            
            # BƯỚC 3: Dùng EasyOCR đọc chữ từ ảnh nhị phân
            ocr_img = cv2.bitwise_not(plate_thresh)
            allowlist = '0123456789ABCDEFGHJKLMNPRSTUVXYZ-'
            ocr_res = reader.readtext(ocr_img, allowlist=allowlist, paragraph=False)
            
            plate_text = ""
            for res in ocr_res:
                plate_text += res[1] + "-"
            plate_text = plate_text.strip("-")
            
            if plate_text:
                cv2.putText(frame, plate_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Đưa ảnh về chung 1 kích thước chiều rộng (ví dụ 400px) để có thể ghép dọc (vstack)
            target_w = 400
            h, w = plate_result.shape[:2]
            target_h = int((target_w / w) * h)
            
            plate_res_resized = cv2.resize(plate_result, (target_w, target_h))
            ocr_img_resized = cv2.resize(ocr_img, (target_w, target_h))
            
            list_plate_results.append(plate_res_resized)
            list_ocr_imgs.append(ocr_img_resized)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Nếu có biển số, ghép tất cả lại thành 1 cột dọc và hiển thị
        if list_plate_results:
            stacked_plates = np.vstack(list_plate_results)
            stacked_ocr = np.vstack(list_ocr_imgs)
            cv2.imshow("1. Danh sach Bien So (Contour)", stacked_plates)
            cv2.imshow("2. Danh sach Anh OCR", stacked_ocr)
        
        cv2.imshow("0. Camera Goc", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
