### 環境安裝

- 下載 python-3.12.11-amd64.exe

- 下載 AppServ    Apache + PHP + MYSQL
Appserv-win32-8.6.0.exe
https://sourceforge.net/projects/appserv/files/AppServ%20Open%20Project/8.6.0/appserv-win32-8.6.0.exe/download

- 下載 YOLOv7
```cmd
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
mkdir weights
```

- 開啟cmd命令提示字元
```cmd
python --version
pip --version
```

- 安裝 YOLOv7 必要套件
```text
matplotlib>=3.2.2
numpy>=1.26           # 放寬到 1.26+（你有 2.0.2，更高 OK）
opencv-python>=4.1.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0         # 你已有 2.8，不會動它
torchvision>=0.8.1
tqdm>=4.41.0
protobuf>=4.25       # 放寬，避免降級
tensorboard>=2.4.1
pandas>=1.1.4
seaborn>=0.11.0
ipython
psutil
thop                 # 新增：YOLOv7 常用
```

- 安裝 PyTorch（Windows）CUDA 12.6
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 寵物監控系統說明
- 架構
```text
pet_monitor_system/
├── 📦 requirements.txt # Python 依賴 
├── ⚙️ manage.py # Django 管理 
├── 🎯 pet_monitor/ # Django 主專案 
├── 📊 monitor/ # 監控應用 (API + 模型) 
├── 🤖 model/ # AI 檢測器 
├── 📺 stream/ # 串流處理 
├── 📄 templates/ # HTML 模板
├──    yolov7 影像辨識
├──    weights 放有yolov7 訓練出來的best.pt 模型權重
├──    venv 虛擬環境(pip 安裝套件)
```


```text
pet_monitor_system/
├── manage.py                  # Django 主入口
├── pet_monitor/               # Django 設定
│   ├── __init__.py
│   ├── settings.py            # 連線 MySQL / 靜態檔案 / REST Framework
│   ├── urls.py                # 前端頁面與 API 路由
│   └── wsgi.py
├── monitor/                   # 行為監控 (API + 模型結果寫入 DB)
│   ├── models.py              # Pet, Behavior
│   ├── views.py               # REST API 實作
│   ├── serializers.py
│   ├── urls.py
│   └── ai_inference.py        # YOLOv7 + OpenCV 推論
├── stream/                    # 串流服務
│   ├── views.py (MJPEG/RTSP)
│   └── urls.py
├── templates/                 # HTML 頁面
│   ├── index.html             # 
│   ├── help.html              # 
│   └── status.html            #
├── yolov7                     # 放 YOLOv7 原始碼
├── db_init.sql                # MySQL 初始化資料
└── start.bat                  # 一鍵啟動 (Windows)
```

- 創建專案資料夾
```cmd
cd C:\Users\你的使用者名稱\Downloads\pet_monitor_mysql57_with_video
```

- 建立虛擬環境，後面指令操作都在這個虛擬環境做
```cmd
python -m venv venv
venv\Scripts\activate
```

- requirements-windows.txt
```text
# ==== Core numeric/science ====
numpy==2.0.2
scipy==1.16.2
pandas==2.2.2
pytz==2025.2
python-dateutil==2.9.0.post0
matplotlib==3.10.0
seaborn==0.13.2

# ==== CV / IO ====
opencv-python==4.12.0.88
Pillow==11.3.0

# ==== YOLOv7 helpers ====
PyYAML==6.0.2
tqdm==4.67.1
tensorboard==2.19.0
protobuf==5.29.5
psutil==5.9.5
thop

# ==== Networking (requests stack) ====
requests==2.32.4
urllib3==2.5.0
chardet==5.2.0
idna==3.10
certifi==2025.8.3

# ==== Torch / TorchVision are installed separately (see above) ====
# (Do NOT pin here to avoid missing CUDA wheels on Windows)

# ==== Extras to satisfy recent torch/torchvision import paths ====
typing_extensions==4.15.0
sympy==1.13.3
packaging==25.0
```

- 在venv 虛擬環境值執行
```cmd
pip install -r requirements-windows.txt
```

### ⚙️ 後端核心設定
- settings.py 連線 MySQL
```python
import pymysql
pymysql.install_as_MySQLdb()

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'pet_monitor',
        'USER': 'root',
        'PASSWORD': '',   # 如果有密碼要改
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {'charset': 'utf8mb4'},
    }
}


INSTALLED_APPS = [
    'rest_framework',
    'monitor',
    'stream',
]
```

- monitor/models.py
```python
from django.db import models

class Pet(models.Model):
    name = models.CharField(max_length=50)

class Behavior(models.Model):
    pet = models.ForeignKey(Pet, on_delete=models.CASCADE)
    behavior = models.CharField(max_length=20)  # eating, toilet, lying
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    duration = models.IntegerField(default=0)  # 秒
```

### 📄 API 路由 (Django REST Framework)
```text
/api/pets/ → 取得寵物資料
/api/behaviors/ → 行為記錄
/api/realtime/ → 即時辨識結果
/api/stream/video/ → 影像串流 (MJPEG)
/api/stream/stop/ → 停止串流
```

### 🐬 MySQL 初始化 (db_init.sql)
```text
CREATE DATABASE IF NOT EXISTS pet_monitor CHARACTER SET utf8mb4;
USE pet_monitor;

CREATE TABLE pet_monitor_pet (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE pet_monitor_behavior (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pet_id INT,
    behavior VARCHAR(20),
    confidence FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration INT DEFAULT 0,
    FOREIGN KEY (pet_id) REFERENCES pet_monitor_pet(id)
);

INSERT INTO pet_monitor_pet (name) VALUES ('小黑'), ('小白');
```



### 建立資料表 & 啟動 Django
```text
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

- 開啟瀏覽器 → http://127.0.0.1:8000/
- 首頁：即時串流、健康預測
- 幫助：快速開始 + API 文件
- 狀態：系統狀態檢查

- 開啟首頁
在瀏覽器打開：
```text
http://127.0.0.1:8000/
```

- 測試 Help 頁
在瀏覽器打開：
```text
http://127.0.0.1:8000/help/
```

- 測試 Status 頁
在瀏覽器打開：
```text
http://127.0.0.1:8000/status/
```

- 測試 API (後端)
即時辨識 (假資料回應)
```cmd
curl http://127.0.0.1:8000/api/realtime/
```
✅ 應該會回：
```json
{
  "current_behavior": "無檢測",
  "confidence": 0.0,
  "health_status": "normal"
}
```

- 測試串流 API（開啟串流）
```cmd
curl http://127.0.0.1:8000/api/stream/video/
```
✅ 預期回應：
```json
{"ok": true}
```

- 測試串流 API（關閉串流）
```cmd
curl http://127.0.0.1:8000/api/stream/stop/
```
✅ 預期回應：
```json
{"stopped": true}
```



- 測試行為紀錄清單
```cmd
curl http://127.0.0.1:8000/api/behaviors/
```
✅ 預期回應：
```json
{"results":[]}
```

- 取得寵物清單
http://127.0.0.1:8000/api/pets/
```json
{"results":[{"id":1,"name":"小黑"}]}
```

測試資料庫
如果你有匯入 db_init.sql 到 MySQL，打開 phpMyAdmin 或 MySQL CLI，執行：
```sql
SELECT * FROM pet_monitor_pet;
```

### Django 3.2 LTS 支援 MySQL 5.7
```text
Django==3.2.25
djangorestframework==3.14.0
PyMySQL==1.1.1
```


# 確認 YOLOv7 + OpenCV 串流程式

前端按下「開始監控」後，實際上應該呼叫 stream/ app 裡的串流邏輯（Django View 或單獨 Python 程式）。
靜態 API → /api/realtime/ 回傳 JSON，沒有真的去讀攝影機 / mp4。

👉 一個 OpenCV 串流程式，例如：
```python
# stream/views.py
from django.http import StreamingHttpResponse
import cv2

def gen_frames():
    cap = cv2.VideoCapture(0)  # 攝影機
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 轉成 JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
```
然後在 urls.py 加一條：
```
path("video_feed/", video_feed, name="video_feed")
```

- 前端 HTML (templates/index.html)
把 <img> 指向後端串流 URL：
```html
<div>
  <h3>即時監控</h3>
  <img id="stream" src="/video_feed/" width="640" height="480" />
</div>
```

- 如果要加 YOLOv7 偵測
在 gen_frames() 裡，把 frame 丟到 YOLOv7 模型跑，畫框後再送回瀏覽器。像這樣：
```python
# 假設已載入 yolo 模型
results = model(frame)  # YOLO 偵測
for *xyxy, conf, cls in results.xyxy[0]:
    label = f"{model.names[int(cls)]} {conf:.2f}"
    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                  (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
```html


### 修改
1. stream/views.py
我會在 gen_frames() 中：
載入 YOLOv7 模型（torch）。
每一幀送進模型推論。
將偵測到的框框與分類（吃飯、上廁所、趴下）畫到 frame 上。
再輸出為 MJPEG。
範例程式片段：
```python
import torch

# 載入模型（請確認 weights/best.pt 存在）
MODEL_PATH = "weights/best.pt"
model = torch.hub.load("WongKinYiu/yolov7", "custom", MODEL_PATH, trust_repo=True)

def gen_frames():
    cap = _open_camera()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 推論
        results = model(frame)
        # 轉換成帶框的 numpy 影像
        frame = results.render()[0]
        # 輸出 JPEG
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            yield (b'--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n' +
                   buffer.tobytes() + b'\\r\\n')
```

- 整合效果
你瀏覽 /video_feed/ → 即時看到鏡頭畫面 + YOLO 偵測框。
偵測到的分類會顯示在框框上（如 eating, toilet, lying）。
best.pt 需放在 weights/ 資料夾底下。

# 進階應用
這份程式會：

載入 YOLOv7 模型（weights/best.pt）。
使用 OpenCV 擷取攝影機影像。
每一幀跑推論並繪製框框與標籤。
透過 Django StreamingHttpResponse 輸出為 MJPEG。
```python
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import numpy as np
import torch
import os

# 攝影機來源：0 = 內建攝影機；也可以改成 rtsp://... 或 http://...
CAMERA_SOURCE = 0

# 模型路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")

# 載入 YOLOv7 模型
print(f"載入 YOLOv7 模型: {MODEL_PATH}")
model = torch.hub.load("yolov7", "custom", path=MODEL_PATH, source="local")

def _open_camera():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        cap.open(CAMERA_SOURCE)
    return cap

def gen_frames():
    cap = _open_camera()
    if not cap or not cap.isOpened():
        blank = (255 * np.ones((240, 320, 3), dtype=np.uint8))
        ok, buffer = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO 推論
            results = model(frame)
            frame = results.render()[0]  # 取帶標註的影像

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    finally:
        cap.release()

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
```

# 重點修復
放置 YOLOv7 原始碼與權重
pet_monitor_system/
├── yolov7/                 ← 這裡放 github 專案（git clone 下來）
│   ├── hubconf.py
│   └── requirements.txt    （內含 numpy<1.24 的相依宣告）
└── weights/
    └── best.pt             ← 你的訓練權重



### weights
- 把 NMS 改在 CPU 上做
- 改 yolov7\utils\general.py 的 non_max_suppression()，在呼叫 NMS 前把資料搬到 CPU：找到這行（大約在函式中部）：
```python
i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
```
改成：
```python
# 在 CPU 上做 NMS（避免 CUDA 版 torchvision::nms 缺失）
i = torchvision.ops.nms(boxes.float().cpu(), scores.cpu(), iou_thres)
```

- 修改 YOLOv7 的 models/experimental.py
打開 yolov7/models/experimental.py，找到：
```python
ckpt = torch.load(w, map_location=map_location)  # load
```
改成:
```python
ckpt = torch.load(w, map_location=map_location, weights_only=False)  # load (PyTorch>=2.6)
```
設 weights_only=False 會回到舊行為；只在你信任權重來源時使用

- **驗證安裝是否真的有 NMS CUDA 核心：**
```python
python - <<PY
import torch, torchvision
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())
print('torchvision', torchvision.__version__)
import torch.ops
print('has torchvision nms:', hasattr(torch.ops.torchvision, 'nms'))
PY
```


### 網頁展示

<img width="886" height="653" alt="截圖 2025-09-28 下午3 57 24" src="https://github.com/user-attachments/assets/f6c8bc15-b0a5-4d9d-bc00-698fa8b20067" />
<img width="880" height="483" alt="截圖 2025-09-28 下午3 59 40" src="https://github.com/user-attachments/assets/218666ff-cc63-4616-a288-8a29af7ac9a5" />
