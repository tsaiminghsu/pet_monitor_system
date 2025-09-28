@echo off
echo 啟動 Django (只跑伺服器，不安裝依賴)...
python manage.py runserver 127.0.0.1:8000
pause
