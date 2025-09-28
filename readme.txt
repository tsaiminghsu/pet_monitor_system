開啟cmd命令提示字元

cd C:\Users\kelvinc0110\Downloads\寵物辨識系統\pet_monitor_mysql57_with_video

#啟動虛擬環境venv
venv\Scripts\activate

python manage.py runserver 127.0.0.1:8000 #啟用網頁

http://127.0.0.1:8000/  #打開瀏覽器開啟首頁

deactivate   #退出虛擬環境venv

===============================================================================

# 測試 API (後端)

cd C:\Users\kelvinc0110\Downloads\寵物辨識系統\pet_monitor_mysql57_with_video

venv\Scripts\activate #啟動虛擬環境venv

#api 在C:\Users\kelvinc0110\Downloads\寵物辨識系統\pet_monitor_mysql57_with_video\pet_monitor\urls.py 有定義

curl http://127.0.0.1:8000/api/pets/
curl http://127.0.0.1:8000/api/behaviors/
curl http://127.0.0.1:8000/api/stream/video/
curl http://127.0.0.1:8000/api/stream/stop/
============ api ============
video_feed/
api/stream/video/
api/realtime/
api/pets/
api/detections/
api/detections/post/
============ api ============ 

curl http://127.0.0.1:8000/api/pets/

{"count": 0, "results": []}


# /api/behaviors 資料量很大這個要小心使用，我只列幾個

curl http://127.0.0.1:8000/api/behaviors/?days=1
{"results": [{"behavior": "0", "confidence": 0.270593, "health_status": "normal", "timestamp": "2025-09-27T07:02:34"}, {"behavior": "0", "confidence": 0.284463, "health_status": "normal", "timestamp": "2025-09-27T07:02:34"}, {"behavior": "0", "confidence": 0.355006, "health_status": "normal", "timestamp": "2025-09-27T07:02:33"}, {"behavior": "0", "confidence": 0.276928, "health_status": "normal", "timestamp": "2025-09-27T07:02:33"}, {"behavior": "0", "confidence": 0.381969, "health_status": "normal", "timestamp": "2025-09-27T07:02:33"}, {"behavior": "0", "confidence": 0.286337, "health_status": "normal", "timestamp": "2025-09-27T07:02:33"}, {"behavior": "dog_eating", "confidence": 0.87, "health_status": "normal", "timestamp": "2025-09-27T06:11:21"}]}



http://127.0.0.1:8000/api/stream/video/?source=0

Warning: Binary output can mess up your terminal. Use
Warning: "--output -" to tell curl to output it to your
Warning: terminal anyway, or consider "--output <FILE>" to
Warning: save to a file.

#如果遇到網頁一直轉圈可能是影像串流無限迴圈把訊號吃掉

taskkill /IM python.exe /F

#再重新執行

python manage.py runserver 127.0.0.1:8000 --noreload --nothreading


==========================================================================================
#資料庫

輸入網址-> localhost
點進這裡 -> phpMyAdmin Database Manager Version 4.6.6

phpMyAdmin
帳號:root
密碼:12345678

資料庫pet_monitor -> 資料表pet_monitor_behavior (其他的不看)

1440秒後資料庫時間到會自動登出要自己再重新登入







gen_frames 串流產生器

















