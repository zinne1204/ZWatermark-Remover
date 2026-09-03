@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  ZWATERMARK REMOVER - NUITKA STANDALONE BUILD
echo ========================================================

python -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [!] Chua cai dat Nuitka. Dang cai dat...
    pip install -U nuitka zstandard
)

echo [1/3] Don dep thu muc build cu...
if exist dist_nuitka rmdir /s /q dist_nuitka

echo [2/3] Bien dich C++ bang Nuitka...
python -m nuitka ^
    --standalone ^
    --windows-console-mode=disable ^
    --enable-plugin=tk-inter ^
    --include-package=customtkinter ^
    --include-package=PIL ^
    --include-package=cv2 ^
    --include-package=onnxruntime ^
    --include-package=tkinterdnd2 ^
    --include-package-data=customtkinter ^
    --include-package-data=tkinterdnd2 ^
    --include-data-dir=assets=assets ^
    --windows-icon-from-ico=assets\app.ico ^
    --output-dir=dist_nuitka ^
    --output-filename=ZWatermarkRemover.exe ^
    --assume-yes-for-downloads ^
    --jobs=4 ^
    app.py

if errorlevel 1 (
    echo [LOI] Bien dich that bai!
    pause
    exit /b 1
)

echo [3/3] Chuan hoa thu muc xuat...
if exist dist_nuitka\app.dist (
    if exist dist_nuitka\ZWatermarkRemover rmdir /s /q dist_nuitka\ZWatermarkRemover
    ren dist_nuitka\app.dist ZWatermarkRemover
)

echo ========================================================
echo [OK] DONG GOI STANDALONE THANH CONG:
echo dist_nuitka\ZWatermarkRemover\ZWatermarkRemover.exe
echo ========================================================
pause
