@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  ZWATERMARK REMOVER - NUITKA ONEFILE BUILD (1 FILE .EXE)
echo ========================================================

python -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [!] Chua cai dat Nuitka. Dang cai dat...
    pip install -U nuitka zstandard
)

echo [1/2] Don dep thu muc build cu...
if exist dist_onefile rmdir /s /q dist_onefile

echo [2/2] Bien dich Onefile bang Nuitka...
python -m nuitka ^
    --onefile ^
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
    --output-dir=dist_onefile ^
    --output-filename=ZWatermarkRemover.exe ^
    --assume-yes-for-downloads ^
    --jobs=4 ^
    app.py

if errorlevel 1 (
    echo [LOI] Bien dich that bai!
    pause
    exit /b 1
)

echo ========================================================
echo [OK] DONG GOI ONEFILE THANH CONG:
echo dist_onefile\ZWatermarkRemover.exe
echo ========================================================
pause
