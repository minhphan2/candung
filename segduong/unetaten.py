import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC ATTENTION U-NET
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

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        avg = self.relu(g1 + x1)
        attention_weights = self.psi(avg)
        return x * attention_weights

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, n=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, n)
        self.enc2 = DoubleConv(n, n*2)
        self.enc3 = DoubleConv(n*2, n*4)
        self.enc4 = DoubleConv(n*4, n*8)
        
        self.bottleneck = DoubleConv(n*8, n*8)

        # Attention Gates
        self.atg1 = AttentionGate(F_g=n*8, F_l=n*8, F_int=n*4)
        self.atg2 = AttentionGate(F_g=n*4, F_l=n*4, F_int=n*2)
        self.atg3 = AttentionGate(F_g=n*2, F_l=n*2, F_int=n)
        self.atg4 = AttentionGate(F_g=n,   F_l=n,   F_int=n//2 if n//2 > 0 else 1)

        # Decoder
        self.dec1 = DoubleConv(n*16, n*4)
        self.dec2 = DoubleConv(n*8, n*2)
        self.dec3 = DoubleConv(n*4, n)
        self.dec4 = DoubleConv(n*2, n)
        
        self.output = nn.Conv2d(n, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x); x = nn.MaxPool2d(2)(enc1)
        enc2 = self.enc2(x); x = nn.MaxPool2d(2)(enc2)
        enc3 = self.enc3(x); x = nn.MaxPool2d(2)(enc3)
        enc4 = self.enc4(x); x = nn.MaxPool2d(2)(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder với Attention
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        attn4 = self.atg1(g=x, x=enc4) 
        x = self.dec1(torch.cat((x, attn4), dim=1))
        
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        attn3 = self.atg2(g=x, x=enc3) 
        x = self.dec2(torch.cat([x, attn3], dim=1))
        
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        attn2 = self.atg3(g=x, x=enc2) 
        x = self.dec3(torch.cat([x, attn2], dim=1))
        
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        attn1 = self.atg4(g=x, x=enc1) 
        x = self.dec4(torch.cat([x, attn1], dim=1))
        
        return torch.sigmoid(self.output(x))

# ==========================================
# 2. KHỞI TẠO HỆ THỐNG
# ==========================================
print("⏳ Đang tải mô hình ATTENTION U-NET (Mặt đường)... (Cơ chế No YOLO)")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load bản Attention U-Net
unet_model = AttentionUNet(in_channels=3, n=16).to(device)
try:
    unet_model.load_state_dict(torch.load("attention_unet_road_model.pt", map_location=device))
    print("✅ Đã load thành công weights attention_unet_road_model.pt!")
except Exception as e:
    print(f"⚠️ Chưa tìm thấy file weights. Lỗi: {e}")

unet_model.eval()

from PIL import Image, ImageEnhance

# Do bước resize và tăng nét đã làm bằng PIL Image ở trong vòng lặp, 
# transform ở đây chỉ chuyển tensor và chuẩn hóa (giống hệt code train mới của cậu)
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. THAM SỐ CẤU HÌNH BUFFER VÀ OCCUPANCY
# ==========================================
baseline_road_pixels = 0 

# Buffer 20 slot
BUFFER_SIZE = 20
FPS = 30
OCCUPANCY_HIGH = 0.4 
CM_THRESHOLD = 0.7   

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
    
    # --- 1. CHẠY ATTENTION U-NET TÌM MẶT ĐƯỜNG HIỆN TẠI ---
    # Chuyển đổi khung hình sang định dạng PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # A. Ép kích thước ảnh về 256x256
    pil_img = pil_img.resize((256, 256))
    
    # B. Dùng đạo hàm làm sắc nét ảnh gấp đôi (Factor = 2.0) để đồng bộ với bộ Data đã train
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(2.0)
    
    # C. Chuẩn hóa thành Tensor và đưa vào Model
    img_tensor = tensor_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = unet_model(img_tensor)
        
    pred_mask = torch.nn.functional.interpolate(pred_mask, size=(h_frame, w_frame), mode='bilinear')
    pred_mask = pred_mask.squeeze().cpu().numpy()
    
    road_mask = (pred_mask > 0.5).astype(np.uint8)
    current_road_pixels = np.sum(road_mask)
    
    # --- 2. TÍNH SỐ PIXEL ĐƯỜNG BỊ MẤT (LOST PIXELS) ---
    if current_road_pixels > baseline_road_pixels:
        baseline_road_pixels = current_road_pixels
        
    lost_road_pixels = baseline_road_pixels - current_road_pixels
    
    if baseline_road_pixels > 0:
        occupancy = float(np.clip(lost_road_pixels / baseline_road_pixels, 0.0, 1.0))
    else:
        occupancy = 0.0
        
    # --- 3. CẬP NHẬT BUFFER (Mỗi 1 giây = 30 frames) ---
    frame_counter += 1
    if frame_counter >= FPS:
        frame_counter = 0
        
        a_new = 1 if occupancy >= OCCUPANCY_HIGH else 0
        a_old = buffer[buffer_idx]
        
        buffer[buffer_idx] = a_new
        buffer_idx = (buffer_idx + 1) % BUFFER_SIZE
        buffer_cm = buffer_cm + (a_new - a_old) / BUFFER_SIZE
        buffer_cm = max(0.0, min(1.0, buffer_cm))
        
    # --- 4. TRỰC QUAN HÓA ---
    display_frame = frame.copy()
    
    # Tô màu vàng cam cho vùng đường được Attention Gate tập trung nhận diện
    display_frame[road_mask == 1] = display_frame[road_mask == 1] * 0.7 + np.array([0, 165, 255]) * 0.3
    
    status = "GRIDLOCK (UN TAC)" if buffer_cm >= CM_THRESHOLD else "FREE FLOW (THONG THOANG)"
    color = (0, 0, 255) if status == "GRIDLOCK (UN TAC)" else (0, 255, 0)
    
    cv2.putText(display_frame, f"Baseline Road Area: {baseline_road_pixels} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Current Road Area : {current_road_pixels} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Lost Road Pixels  : {lost_road_pixels} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(display_frame, f"Occupancy (O)     : {occupancy*100:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    buffer_str = "[" + ", ".join(map(str, buffer)) + "]"
    cv2.putText(display_frame, f"Buffer (20s): {buffer_str}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display_frame, f"Buffer CM   : {buffer_cm:.2f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.putText(display_frame, f"STATUS: {status}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("ATTENTION UNet Traffic Monitor (No YOLO)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
