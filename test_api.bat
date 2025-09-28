@echo off
echo 測試 Django API 中...
echo.

echo [1] 測試 /api/realtime/
curl http://127.0.0.1:8000/api/realtime/
echo.

echo [2] 測試 /api/pets/
curl http://127.0.0.1:8000/api/pets/
echo.

echo [3] 測試 /api/behaviors/
curl http://127.0.0.1:8000/api/behaviors/
echo.

echo [4] 測試 /api/behaviors/weekly_summary/
curl http://127.0.0.1:8000/api/behaviors/weekly_summary/
echo.

echo [5] 測試 /api/stream/video/
curl http://127.0.0.1:8000/api/stream/video/
echo.

echo [6] 測試 /api/stream/stop/
curl http://127.0.0.1:8000/api/stream/stop/
echo.

echo 測試完成！
pause
