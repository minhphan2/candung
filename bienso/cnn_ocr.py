import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Kiến trúc mạng CNN (giống hệt lúc bạn train)
class CharCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CharCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNNOCR:
    def __init__(self, weight_path="best_char_cnn_softmax.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharCNN(num_classes=36).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # 36 classes tương ứng với 0-9 và A-Z theo thứ tự bảng chữ cái
        self.classes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def predict_char(self, char_img):
        """Đưa ảnh 1 ký tự qua model để dự đoán"""
        tensor = self.transform(char_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes[predicted.item()]

    def recognize_plate(self, plate_bgr):
        """Tiền xử lý, phân tách ký tự (Segmentation) và đọc chữ theo chuẩn Mayfest2023"""
        # 1. Tiền xử lý ảnh (HSV -> Kênh V -> Adaptive Threshold)
        hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
        V = cv2.split(hsv)[2]
        
        # Nhị phân hóa động (Adaptive Threshold) với block_size lớn để lọc sáng/tối
        thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
        
        # Đảo ngược: Chữ thành Trắng, Nền thành Đen để tìm contours và connected components
        thresh = cv2.bitwise_not(thresh)
        
        # 2. Lọc nhiễu bằng Connected Components
        num_labels, labels = cv2.connectedComponents(thresh)
        mask = np.zeros(thresh.shape, dtype="uint8")
        
        total_pixels = thresh.shape[0] * thresh.shape[1]
        lower = total_pixels // 120
        upper = total_pixels // 20
        
        for label in range(1, num_labels):
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            
            # Chỉ giữ lại các "cục" màu trắng có diện tích vừa phải (không phải hạt bụi, không phải viền biển to đùng)
            if lower < numPixels < upper:
                mask = cv2.add(mask, labelMask)
                
        # 3. Tìm Bounding Box của từng ký tự trên ảnh mask đã sạch nhiễu
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        
        if not boundingBoxes:
            return ""
            
        # Lọc thêm 1 bước nữa dựa trên kích thước trung bình để bỏ các nét đứt
        arr = np.array(boundingBoxes)
        mean_w = np.mean(arr[:, 2])
        mean_h = np.mean(arr[:, 3])
        threshold_w = mean_w * 1.5
        threshold_h = mean_h * 1.5
        
        valid_boxes = []
        for box in boundingBoxes:
            x, y, w, h = box
            if w < threshold_w and h < threshold_h and h > 10: # Giữ lại các box không quá to và không quá bẹp
                valid_boxes.append(box)
                
        if not valid_boxes:
            return ""
            
        # 4. Phân tách dòng và sắp xếp
        arr_valid = np.array(valid_boxes)
        avg_height = np.mean(arr_valid[:, 3])
        y_diff = max(arr_valid[:, 1]) - min(arr_valid[:, 1])
        
        # Nếu khoảng cách Y giữa ký tự cao nhất và thấp nhất lớn -> Biển 2 dòng
        if y_diff > 0.45 * avg_height:
            y_thresh = (min(arr_valid[:, 1]) + max(arr_valid[:, 1])) / 2
            line1 = sorted([b for b in valid_boxes if b[1] <= y_thresh], key=lambda b: b[0])
            line2 = sorted([b for b in valid_boxes if b[1] > y_thresh], key=lambda b: b[0])
            sorted_boxes = line1 + [{"text": "-"}] + line2
        else:
            sorted_boxes = sorted(valid_boxes, key=lambda b: b[0])
            
        # 5. Cắt từng ký tự và đưa vào CNN
        result = ""
        for item in sorted_boxes:
            if isinstance(item, dict) and item.get("text") == "-":
                result += "-"
                continue
                
            x, y, w, h = item
            # Cắt ký tự từ lớp mask (lúc này chữ là trắng, nền đen)
            # Khác với code cũ cắt từ ảnh gốc, cắt từ mask sẽ SẠCH TINH TƯƠM 100% không còn một hạt bụi nào!
            character = mask[y:y+h, x:x+w]
            
            # Nghịch đảo lại: Chữ Đen, Nền Trắng (chuẩn form của dataset CNN)
            character = cv2.bitwise_not(character)
            
            # Padding thành hình vuông
            rows, columns = character.shape
            side = max(rows, columns)
            pad_h = (side - rows) // 2
            pad_w = (side - columns) // 2
            
            # Nền đang là trắng (255) -> padding màu trắng
            char_padded = cv2.copyMakeBorder(character, pad_h+2, pad_h+2, pad_w+2, pad_w+2, cv2.BORDER_CONSTANT, value=255)
            
            # Đoán ký tự bằng mô hình CNN của bạn
            char_pred = self.predict_char(char_padded)
            result += char_pred
            
        return result
