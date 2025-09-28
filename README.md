#環境檢查

- 下載 python-3.12.11-amd64.exe
Python 3.12.11

- 下載 AppServ    Apache + PHP + MYSQL
Appserv-win32-8.6.0.exe
https://sourceforge.net/projects/appserv/files/AppServ%20Open%20Project/8.6.0/appserv-win32-8.6.0.exe/download

- 開啟cmd命令提示字元
```cmd
python --version
pip --version
```


### 安裝套裝

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



cd C:\Users\你的使用者名稱\Downloads\pet_monitor_mysql57_with_video

# 建立虛擬環境，有看到venv資料夾就不要再操作這個步驟

```cmd
python -m venv venv
venv\Scripts\activate
```

### 架構

```text
pet_monitor_system
├── 📦 requirements.txt # Python 依賴 
├── ⚙️ manage.py # Django 管理 
├── 🎯 pet_monitor/ # Django 主專案 
├── 📊 monitor/ # 監控應用 (API + 模型) 
├── 🤖 model/ # AI 檢測器 
├── 📺 stream/ # 串流處理 
├── 📄 templates/ # HTML 模板
```

### 網頁展示

<img width="886" height="653" alt="截圖 2025-09-28 下午3 57 24" src="https://github.com/user-attachments/assets/f6c8bc15-b0a5-4d9d-bc00-698fa8b20067" />
<img width="880" height="483" alt="截圖 2025-09-28 下午3 59 40" src="https://github.com/user-attachments/assets/218666ff-cc63-4616-a288-8a29af7ac9a5" />
