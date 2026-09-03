"""
Quản lý tập trung đường dẫn tài nguyên, binary (FFmpeg/FFprobe) và model AI cho ZWatermark Remover.
Hỗ trợ chạy mượt mà trên cả:
  1. Chạy mã nguồn trực tiếp (python app.py)
  2. Bản đóng gói Nuitka Standalone
  3. Bản đóng gói Nuitka Onefile
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    """Kiểm tra xem ứng dụng có đang chạy ở dạng đóng gói hay không."""
    return bool(
        getattr(sys, "frozen", False)
        or globals().get("__compiled__", False)
        or getattr(sys, "__compiled__", False)
    )


def get_app_dir() -> Path:
    """Lấy thư mục gốc của ứng dụng an toàn trên mọi môi trường."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_asset_path(name: str) -> Optional[Path]:
    """Tìm đường dẫn file tài nguyên trong thư mục assets."""
    app_dir = get_app_dir()
    candidates = [
        app_dir / "assets" / name,
        app_dir / name,
        Path(sys.executable).resolve().parent / "assets" / name,
        Path(sys.executable).resolve().parent / name,
        Path.cwd() / "assets" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def get_ffmpeg_path() -> str:
    """Tìm đường dẫn thực thi của ffmpeg (ưu tiên bản đóng gói đi kèm -> PATH hệ thống)."""
    app_dir = get_app_dir()
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidates = [
        app_dir / exe_name,
        Path(sys.executable).resolve().parent / exe_name,
        app_dir / "assets" / exe_name,
        app_dir / "assets" / "bin" / exe_name,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def get_ffprobe_path() -> str:
    """Tìm đường dẫn thực thi của ffprobe (ưu tiên bản đóng gói đi kèm -> PATH hệ thống)."""
    app_dir = get_app_dir()
    exe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    candidates = [
        app_dir / exe_name,
        Path(sys.executable).resolve().parent / exe_name,
        app_dir / "assets" / exe_name,
        app_dir / "assets" / "bin" / exe_name,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return shutil.which("ffprobe") or "ffprobe"


def get_model_path(model_name: str = "lama_fp32.onnx") -> Path:
    """Lấy đường dẫn model AI inpainting (ưu tiên assets offline -> thư mục cache người dùng)."""
    # 1. Kiểm tra nếu có sẵn trong assets hoặc cạnh file exe (chạy offline không cần mạng)
    local_asset = get_asset_path(model_name)
    if local_asset and local_asset.is_file() and local_asset.stat().st_size > 1_000_000:
        return local_asset
    
    cand = get_app_dir() / "assets" / "models" / model_name
    if cand.is_file() and cand.stat().st_size > 1_000_000:
        return cand

    # 2. Thư mục cache chuẩn người dùng
    cache_dir = Path.home() / ".cache" / "zwatermark"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / model_name
