@echo off
echo ========================================================
echo  Training YOLO Fauna XR dengan NVIDIA GeForce RTX 3050
echo ========================================================
cd /d "%~dp0"
.\.venv\Scripts\python.exe track\train_model.py --epochs 50 --batch 8 --imgsz 320
pause
