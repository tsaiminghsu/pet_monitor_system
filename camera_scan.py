import cv2
import torch
from django.http import StreamingHttpResponse
from django.shortcuts import render
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.torch_utils import select_device

# ========== YOLOv7 初始化 ==========
WEIGHTS = "weights/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

model = attempt_load(WEIGHTS, map_location=device)
model.eval()
print("✅ YOLOv7 模型載入完成")

# ========== 相機掃描 ==========
def scan_cameras(max_cams=5):
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available

# ========== 即時串流產生器 ==========
def gen_frames(cam_id=0):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"❌ 無法開啟相機 {cam_id}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- YOLOv7 推論 ---
        img = torch.from_numpy(frame).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred.tolist():
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"{int(cls)} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

# ========== Django 視圖 ==========
def index(request):
    return render(request, "index.html")

def video_feed(request, cam_id=0):
    return StreamingHttpResponse(
        gen_frames(cam_id),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

def camera_list(request):
    cameras = scan_cameras()
    return render(request, "camera_list.html", {"cameras": cameras})
