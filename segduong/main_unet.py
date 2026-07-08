import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC U-NET
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),   
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n=16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, n)
        self.enc2 = DoubleConv(n, n*2)
        self.enc3 = DoubleConv(n*2, n*4)
        self.enc4 = DoubleConv(n*4, n*8)
        self.bottleneck = DoubleConv(n*8, n*8)
        self.dec1 = DoubleConv(n*16, n*4)
        self.dec2 = DoubleConv(n*8, n*2)
        self.dec3 = DoubleConv(n*4, n)
        self.dec4 = DoubleConv(n*2, n)
        self.output = nn.Conv2d(n, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x); x = nn.MaxPool2d(2)(enc1)
        enc2 = self.enc2(x); x = nn.MaxPool2d(2)(enc2)
        enc3 = self.enc3(x); x = nn.MaxPool2d(2)(enc3)
        enc4 = self.enc4(x); x = nn.MaxPool2d(2)(enc4)
        x = self.bottleneck(x)
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.dec1(torch.cat((x, enc4), dim=1))
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.dec2(torch.cat([x, enc3], dim=1))
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.dec3(torch.cat([x, enc2], dim=1))
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.dec4(torch.cat([x, enc1], dim=1))
        return torch.sigmoid(self.output(x))

# ==========================================
# 2. KHỞI TẠO HỆ THỐNG
# ==========================================
print("⏳ Đang tải mô hình Custom UNet (Mặt đường)... (ĐÃ TẮT YOLO THEO YÊU CẦU)")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

unet_model = UNet(in_channels=3, n=16).to(device)
unet_model.load_state_dict(torch.load("unet_road_model_v2 (1).pt", map_location=device))
unet_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. THAM SỐ CẤU HÌNH BUFFER VÀ OCCUPANCY
# ==========================================
# Đường cơ sở (Lưu diện tích đường lớn nhất từng thấy khi đường trống)
baseline_road_pixels = 0 

# Buffer 20 slot (Lấy mẫu mỗi giây)
BUFFER_SIZE = 20
FPS = 30 # Giả định video 30 hình/giây
OCCUPANCY_HIGH = 0.4 # Ngưỡng diện tích đường bị mất để tính là 1
CM_THRESHOLD = 0.7   # Nếu >70% số slot trong buffer là 1 -> ÙN TẮC

buffer = [0] * BUFFER_SIZE
buffer_idx = 0
buffer_cm = 0.0
frame_counter = 0

video_path = "4K Road traffic video for object detection and tracking.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (800, 450))
    h_frame, w_frame = frame.shape[:2]
    
    # --- 1. CHẠY U-NET TÌM MẶT ĐƯỜNG HIỆN TẠI ---
    img_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = unet_model(img_tensor)
        
    pred_mask = torch.nn.functional.interpolate(pred_mask, size=(h_frame, w_frame), mode='bilinear')
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    # Lấy ra các pixel đang được AI nhìn thấy là đường
    road_mask = (pred_mask > 0.5).astype(np.uint8)
    current_road_pixels = np.sum(road_mask)
    
    # --- 2. TÍNH SỐ PIXEL ĐƯỜNG BỊ MẤT (LOST PIXELS) ---
    # Tự động học "Diện tích đường tối đa" (Khi đường vắng nhất)
    if current_road_pixels > baseline_road_pixels:
        baseline_road_pixels = current_road_pixels
        
    # Số pixel đường bị mất = Diện tích đường tối đa - Diện tích đường hiện tại đang thấy
    lost_road_pixels = baseline_road_pixels - current_road_pixels
    
    if baseline_road_pixels > 0:
        occupancy = float(np.clip(lost_road_pixels / baseline_road_pixels, 0.0, 1.0))
    else:
        occupancy = 0.0
        
    # --- 3. CẬP NHẬT BUFFER (Mỗi 1 giây = 30 frames) ---
    frame_counter += 1
    if frame_counter >= FPS:
        frame_counter = 0
        
        # Nếu Occupancy vượt ngưỡng thì gán bằng 1 (tắc), ngược lại là 0 (thoáng)
        a_new = 1 if occupancy >= OCCUPANCY_HIGH else 0
        a_old = buffer[buffer_idx]
        
        # Đưa vào buffer và tính lại CM (Current Mean)
        buffer[buffer_idx] = a_new
        buffer_idx = (buffer_idx + 1) % BUFFER_SIZE
        buffer_cm = buffer_cm + (a_new - a_old) / BUFFER_SIZE
        buffer_cm = max(0.0, min(1.0, buffer_cm))
        
    # --- 4. TRỰC QUAN HÓA (VISUALIZATION) ---
    display_frame = frame.copy()
    
    # Tô màu xanh lam cho vùng đường hiện tại
    display_frame[road_mask == 1] = display_frame[road_mask == 1] * 0.7 + np.array([255, 0, 0]) * 0.3
    
    # Xác định trạng thái
    status = "GRIDLOCK (UN TAC)" if buffer_cm >= CM_THRESHOLD else "FREE FLOW (THONG THOANG)"
    color = (0, 0, 255) if status == "GRIDLOCK (UN TAC)" else (0, 255, 0)
    
    # Vẽ Dashboard
    cv2.putText(display_frame, f"Baseline Road Area: {baseline_road_pixels} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Current Road Area : {current_road_pixels} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Lost Road Pixels  : {lost_road_pixels} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(display_frame, f"Occupancy (O)     : {occupancy*100:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Vẽ mảng Buffer
    buffer_str = "[" + ", ".join(map(str, buffer)) + "]"
    cv2.putText(display_frame, f"Buffer (20s): {buffer_str}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display_frame, f"Buffer CM   : {buffer_cm:.2f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.putText(display_frame, f"STATUS: {status}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("UNet Traffic Monitor (No YOLO)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
