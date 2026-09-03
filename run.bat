@echo off
title ZWatermark Remover
python app.py
if errorlevel 1 (
    echo.
    echo [LOI] Khong the chay app.py. Kiem tra xem da cai thu vien chua:
    echo pip install -r requirements.txt
    pause
)
