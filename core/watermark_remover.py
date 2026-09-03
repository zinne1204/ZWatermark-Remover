"""Xóa watermark sparkle Gemini & Veo video — Tối ưu Memory Pipeline & Hardware Acceleration."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .paths import get_app_dir, get_ffmpeg_path, get_ffprobe_path, get_model_path

LogFn = Callable[[str], None]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

_MASK_CACHE: Dict[Tuple[int, float], np.ndarray] = {}


def _astroid_mask(size: int, power: float = 2 / 3) -> np.ndarray:
    n = max(8, int(size))
    key = (n, round(float(power), 4))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    samples = 4
    sx = (np.arange(samples) + 0.5) / samples
    sy = (np.arange(samples) + 0.5) / samples
    sx_g = sx.reshape(1, 1, 1, samples)
    sy_g = sy.reshape(1, 1, samples, 1)
    px = (np.arange(n).reshape(n, 1, 1, 1) + sx_g) / n * 2 - 1
    py = (np.arange(n).reshape(1, n, 1, 1) + sy_g) / n * 2 - 1
    inside = (np.abs(px) ** power + np.abs(py) ** power) <= 1.0
    data = inside.mean(axis=(2, 3)).astype(np.float32)
    _MASK_CACHE[key] = data
    return data


def _luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _ncc_fast(patch: np.ndarray, mask: np.ndarray, mask_c: np.ndarray, mask_norm: float) -> float:
    a = patch.astype(np.float64).ravel()
    a = a - a.mean()
    denom = np.sqrt((a * a).sum()) * mask_norm
    if denom < 1e-9:
        return -1.0
    return float(np.dot(a, mask_c) / denom)


def _prep_mask(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    m = mask.astype(np.float64).ravel()
    mc = m - m.mean()
    norm = float(np.sqrt(np.dot(mc, mc)))
    return mc, max(norm, 1e-9)


def locate_sparkle(
    rgb: np.ndarray,
    ncc_threshold: float = 0.42,
) -> Optional[Tuple[int, int, int, float, np.ndarray]]:
    h, w = rgb.shape[:2]
    short = min(h, w)
    luma = _luma(rgb)
    best: Optional[Tuple[float, int, int, int, np.ndarray]] = None

    size_ratios = (0.052, 0.0586, 0.0625, 0.0664, 0.0703, 0.0742, 0.082)
    margin_ratios = (0.055, 0.07, 0.091, 0.11)
    powers = (2 / 3, 0.62)

    for ratio in size_ratios:
        size = max(20, int(round(short * ratio)))
        if size >= w or size >= h:
            continue
        for power in powers:
            mask = _astroid_mask(size, power)
            mc, mn = _prep_mask(mask)
            slack = max(2, int(round(short * 0.016)))
            step = 2 if size >= 40 else 1

            anchors = []
            for mr in margin_ratios:
                m = int(round(short * mr))
                anchors.append((w - size - m, h - size - m))
            anchors.append((w // 2 - size // 2, h - size - int(short * 0.08)))

            for ax, ay in anchors:
                for dy in range(-slack, slack + 1, step):
                    for dx in range(-slack, slack + 1, step):
                        x, y = ax + dx, ay + dy
                        if x < 0 or y < 0 or x + size > w or y + size > h:
                            continue
                        score = _ncc_fast(
                            luma[y : y + size, x : x + size], mask, mc, mn
                        )
                        if best is None or score > best[0]:
                            best = (score, x, y, size, mask)

    if best is None or best[0] < ncc_threshold:
        for size in (40, 48, 56, 64, 72):
            if size >= w or size >= h:
                continue
            mask = _astroid_mask(size)
            mc, mn = _prep_mask(mask)
            for margin in (48, 64, 80, 96):
                ax, ay = w - size - margin, h - size - margin
                for dy in range(-6, 7, 2):
                    for dx in range(-6, 7, 2):
                        x, y = ax + dx, ay + dy
                        if x < 0 or y < 0 or x + size > w or y + size > h:
                            continue
                        score = _ncc_fast(
                            luma[y : y + size, x : x + size], mask, mc, mn
                        )
                        if best is None or score > best[0]:
                            best = (score, x, y, size, mask)

    if best is None or best[0] < ncc_threshold:
        return None
    score, x, y, size, mask = best
    return x, y, size, score, mask


def _estimate_gain(rgb: np.ndarray, mask: np.ndarray, x: int, y: int) -> float:
    luma = _luma(rgb)
    size = mask.shape[0]
    patch = luma[y : y + size, x : x + size]
    mc, mn = _prep_mask(mask)
    best_g, best_abs = 0.45, 1e9
    for g in np.linspace(0.2, 0.72, 11):
        alpha = np.clip(mask * float(g), 0, 0.92)
        inv = np.maximum(1.0 - alpha, 1e-6)
        corrected = np.clip((patch - 255.0 * alpha) / inv, 0, 255)
        score = abs(_ncc_fast(corrected, mask, mc, mn))
        if score < best_abs:
            best_abs, best_g = score, float(g)
    return best_g


def _simple_inpaint(rgb: np.ndarray, mask_full: np.ndarray, radius: int = 5) -> np.ndarray:
    try:
        from scipy import ndimage

        out = rgb.astype(np.float32)
        inv = (mask_full == 0).astype(np.float32)
        for c in range(3):
            ch = out[..., c]
            num = ndimage.uniform_filter(ch * inv, size=radius * 2 + 1)
            den = ndimage.uniform_filter(inv, size=radius * 2 + 1)
            fill = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
            ch = np.where(mask_full > 0, fill, ch)
            out[..., c] = ch
        return np.clip(out, 0, 255).astype(np.uint8)
    except ImportError:
        out = rgb.astype(np.float64).copy()
        h, w = mask_full.shape
        ys, xs = np.where(mask_full > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1 = max(0, y - radius), min(h, y + radius + 1)
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            block = out[y0:y1, x0:x1]
            good = mask_full[y0:y1, x0:x1] == 0
            if not np.any(good):
                continue
            out[y, x] = block[good].mean(axis=0)
        return np.clip(out, 0, 255).astype(np.uint8)


def remove_sparkle(rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    found = locate_sparkle(rgb)
    if not found:
        return rgb.copy(), {"applied": False, "reason": "NOT_FOUND"}

    x, y, size, ncc, mask = found
    out = rgb.copy()
    gain = _estimate_gain(out, mask, x, y)

    alpha = np.clip(mask * gain, 0, 0.92)[..., None]
    region = out[y : y + size, x : x + size].astype(np.float32)
    inv = np.maximum(1.0 - alpha, 1e-6)
    region = np.clip((region - 255.0 * alpha) / inv, 0, 255)
    out[y : y + size, x : x + size] = region.astype(np.uint8)

    body = (mask >= 0.04).astype(np.uint8)
    full = np.zeros(out.shape[:2], dtype=np.uint8)
    full[y : y + size, x : x + size] = body

    engine = "numpy"
    try:
        import cv2

        rad = max(3, size // 12)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))
        full_d = cv2.dilate(full * 255, k, iterations=2)
        a = cv2.inpaint(out, full_d, max(3, rad), cv2.INPAINT_TELEA)
        b = cv2.inpaint(out, full_d, max(3, rad), cv2.INPAINT_NS)
        m = (full_d > 0).astype(np.float32)[..., None]
        out = (a.astype(np.float32) * (1 - m * 0.5) + b.astype(np.float32) * (m * 0.5))
        out = np.clip(out, 0, 255).astype(np.uint8)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.morphologyEx(full_d, cv2.MORPH_GRADIENT, k2)
        if int(edge.sum()) > 0:
            out = cv2.inpaint(out, edge, 2, cv2.INPAINT_TELEA)
        engine = "opencv"
    except ImportError:
        h, w = full.shape
        pad = max(3, size // 10)
        ys, xs = np.where(full > 0)
        full_d = full.copy()
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            full_d[
                max(0, yy - pad) : min(h, yy + pad + 1),
                max(0, xx - pad) : min(w, xx + pad + 1),
            ] = 1
        out = _simple_inpaint(out, full_d, radius=pad + 2)

    return out, {
        "applied": True,
        "ncc": round(float(ncc), 4),
        "gain": round(float(gain), 3),
        "pos": {"x": int(x), "y": int(y), "size": int(size)},
        "engine": engine,
    }


def remove_corner_logo(rgb: np.ndarray, corner: str = "br") -> tuple:
    h, w = rgb.shape[:2]
    bw = max(48, int(w * 0.16))
    bh = max(28, int(h * 0.10))
    pad = max(4, int(min(h, w) * 0.012))
    if corner == "br":
        x0, y0 = w - bw - pad, h - bh - pad
    elif corner == "bl":
        x0, y0 = pad, h - bh - pad
    elif corner == "tr":
        x0, y0 = w - bw - pad, pad
    else:
        x0, y0 = pad, pad
    x0, y0 = max(0, x0), max(0, y0)

    vw, vh = max(32, int(w * 0.07)), max(16, int(h * 0.045))
    lx0, ly0 = w - vw - pad, h - vh - pad
    lx0, ly0 = max(0, lx0), max(0, ly0)
    patch = rgb[ly0:h - pad, lx0:w - pad]
    luma = _luma(patch)
    med = float(np.median(luma)) if luma.size else 128.0
    thr = max(med + 20.0, 170.0)
    local = (luma >= thr).astype(np.uint8)
    if local.mean() < 0.01:
        local[:] = 1

    full = np.zeros((h, w), dtype=np.uint8)
    full[ly0:h - pad, lx0:w - pad] = local

    out = rgb.copy()
    engine = "numpy"
    try:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        full_d = cv2.dilate(full * 255, k, iterations=1)
        painted = cv2.inpaint(out, full_d, 2, cv2.INPAINT_TELEA)
        m = (full_d > 0)[..., None]
        out = np.where(m, painted, out)
        engine = "opencv-thin"
    except ImportError:
        out = _simple_inpaint(out, full, radius=3)

    return out, {
        "applied": True,
        "kind": "corner_logo",
        "engine": engine,
        "box": {"x": int(lx0), "y": int(ly0), "w": int(w - pad - lx0), "h": int(h - pad - ly0)},
    }


def inpaint_box(rgb: np.ndarray, box: dict, use_ai: bool = False, ai_passes: int = 2, prefer_gpu: bool = False) -> tuple:
    h, w = rgb.shape[:2]
    x = max(0, int(box.get("x", 0)))
    y = max(0, int(box.get("y", 0)))
    bw = max(1, int(box.get("w", 1)))
    bh = max(1, int(box.get("h", 1)))
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x or y1 <= y:
        return rgb.copy(), {"applied": False, "reason": "BAD_BOX"}

    pad_x = max(1, int(bw * 0.06))
    pad_y = max(1, int(bh * 0.08))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)

    patch = rgb[y0:y1, x0:x1]
    luma = _luma(patch)
    if use_ai:
        local = np.ones(luma.shape, dtype=np.uint8)
    else:
        med = float(np.median(luma))
        thr = max(med + 12.0, float(np.percentile(luma, 60)))
        local = (luma >= thr).astype(np.uint8)
        if local.mean() < 0.04:
            local[:] = 1

    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:y1, x0:x1] = local

    out = rgb.copy()
    engine = "neighbor"
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        full_d = cv2.dilate(full * 255, k, iterations=1)
    except Exception:
        full_d = full * 255
        cv2 = None

    if use_ai:
        out, engine = ai_inpaint(out, full_d, passes=ai_passes, prefer_gpu=prefer_gpu)
    elif cv2 is not None:
        painted = cv2.inpaint(out, full_d, 2, cv2.INPAINT_TELEA)
        m = (full_d > 0)[..., None]
        out = np.where(m, painted, out)
        engine = "opencv-thin"
    else:
        out = _simple_inpaint(out, (full_d > 0).astype(np.uint8), radius=3)

    return out, {
        "applied": True,
        "kind": "manual_box",
        "engine": engine,
        "box": {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)},
    }


def remove_watermark(rgb: np.ndarray, prefer_corner: bool = False) -> tuple:
    if prefer_corner:
        out, info = remove_corner_logo(rgb)
        spark, sinfo = remove_sparkle(out)
        if sinfo.get("applied"):
            sinfo["kind"] = "sparkle+corner"
            sinfo["corner"] = True
            return spark, sinfo
        return out, info
    out, info = remove_sparkle(rgb)
    if info.get("applied"):
        info["kind"] = "sparkle"
        return out, info
    return remove_corner_logo(rgb)


def process_file(
    src: Path,
    dst: Path,
    log: Optional[LogFn] = None,
    use_ai: bool = False,
    box: Optional[dict] = None,
    mode: str = "auto",
    custom_mask: Optional[np.ndarray] = None,
) -> dict:
    from PIL import Image

    def _log(m: str) -> None:
        if log:
            log(m)

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    rgb = np.asarray(img)

    if custom_mask is not None and int(custom_mask.sum()) > 0:
        out, info = inpaint_mask(rgb, custom_mask, use_ai=use_ai)
    elif box:
        out, info = inpaint_box(rgb, box, use_ai=use_ai, ai_passes=1 if use_ai else 2)
    elif mode == "stock_text":
        out, info = remove_stock_text_watermark(rgb, use_ai=use_ai)
        if not info.get("applied"):
            out, info = remove_watermark(rgb, prefer_corner=False)
    elif mode == "horizontal_text":
        out, info = remove_horizontal_text_watermark(rgb, use_ai=use_ai)
    else:
        out, info = remove_smart_auto_watermark(rgb, use_ai=use_ai)

    Image.fromarray(out).save(dst, quality=95)

    info["src"] = str(src)
    info["dst"] = str(dst)
    if info.get("applied"):
        _log(
            f"✓ {src.name} → xóa ({info.get('kind', 'watermark')}, {info.get('engine', 'ai')})"
        )
    else:
        _log(f"· {src.name} → không thấy watermark")
    return info


def clean_media_file(
    src: Path | str,
    output_dir: Optional[Path | str] = None,
    log: Optional[LogFn] = None,
    use_ai: bool = True,
    box: Optional[dict] = None,
) -> Path:
    """Tự động xóa watermark bằng AI Inpainting cho ảnh hoặc video.
    
    - Tự động nhận diện định dạng ảnh / video.
    - Lưu file sạch vào output_dir (mặc định: <thư_mục_chứa_src>/_cleaned_wm).
    - Có cơ chế cache thông minh: tái sử dụng file sạch nếu đã tồn tại và mới hơn file gốc.
    - An toàn: fallback về file gốc nếu gặp lỗi xử lý.
    """
    def _log(m: str) -> None:
        if log:
            log(m)

    src_p = Path(src)
    if not src_p.is_file():
        return src_p

    ext = src_p.suffix.lower()
    dst_dir = Path(output_dir) if output_dir else src_p.parent / "_cleaned_wm"
    dst_dir.mkdir(parents=True, exist_ok=True)

    if ext in IMAGE_EXT:
        dst = dst_dir / f"{src_p.stem}_clean{src_p.suffix}"
        if dst.is_file() and dst.stat().st_size > 0 and dst.stat().st_mtime >= src_p.stat().st_mtime:
            _log(f"  [CACHE] Tái sử dụng ảnh sạch: {dst.name}")
            return dst

        try:
            info = process_file(src_p, dst, log=log, use_ai=use_ai, box=box)
            if dst.is_file() and dst.stat().st_size > 0:
                return dst
        except Exception as e:
            _log(f"  [LỖI XỬ LÝ ẢNH] {src_p.name}: {e} -> Giữ nguyên file gốc")
            return src_p

    elif ext in VIDEO_EXT:
        dst = dst_dir / f"{src_p.stem}_clean.mp4"
        if dst.is_file() and dst.stat().st_size > 0 and dst.stat().st_mtime >= src_p.stat().st_mtime:
            _log(f"  [CACHE] Tái sử dụng video sạch: {dst.name}")
            return dst

        try:
            info = process_video(src_p, dst, log=log, max_seconds=600.0, box=box, use_ai=use_ai)
            if dst.is_file() and dst.stat().st_size > 0:
                return dst
        except Exception as e:
            _log(f"  [LỖI XỬ LÝ VIDEO] {src_p.name}: {e} -> Giữ nguyên file gốc")
            return src_p

    return src_p


def process_folder(
    folder: Path, out_dir: Path, log: Optional[LogFn] = None
) -> List[dict]:
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )
    if not files:
        raise ValueError(f"Không có ảnh trong: {folder}")
    return [process_file(p, out_dir / p.name, log=log) for p in files]


_LAMA = None
_LAMA_ERR = None


def lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except Exception:
        return False


def _get_lama():
    global _LAMA, _LAMA_ERR
    if _LAMA is not None:
        return _LAMA
    if _LAMA_ERR is not None:
        raise RuntimeError(_LAMA_ERR)
    try:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
        return _LAMA
    except Exception as e:
        _LAMA_ERR = (
            "Chưa cài AI LaMa. Chạy:\n"
            "pip install simple-lama-inpainting torch pillow\n"
            f"Chi tiết: {e}"
        )
        raise RuntimeError(_LAMA_ERR) from e


def _lama_inpaint(rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    from PIL import Image

    lama = _get_lama()
    m = mask_u8
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    else:
        m = m.astype(np.uint8)
        m = np.where(m > 0, 255, 0).astype(np.uint8)
    img = Image.fromarray(rgb)
    mask = Image.fromarray(m).convert("L")
    out = lama(img, mask)
    if hasattr(out, "convert"):
        out = out.convert("RGB")
        return np.asarray(out)
    return np.asarray(out)


LAMA_ONNX_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
_LAMA_ONNX_SESS = None


def _lama_onnx_path() -> Path:
    return get_model_path("lama_fp32.onnx")


def lama_onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_lama_onnx(log: Optional[LogFn] = None) -> Path:
    import urllib.request

    dest = _lama_onnx_path()
    if dest.is_file() and dest.stat().st_size > 10_000_000:
        return dest
    # Kiểm tra trong assets/models hoặc scratch
    for candidate in [
        Path(get_app_dir()) / "assets" / "models" / "lama_fp32.onnx",
        Path(__file__).resolve().parent.parent / "assets" / "models" / "lama_fp32.onnx",
        Path("scratch") / "lama_fp32.onnx",
    ]:
        if candidate.is_file() and candidate.stat().st_size > 10_000_000:
            return candidate
    if log:
        log("Đang tải model LaMa AI (~200MB) cho chất lượng xóa ảnh đỉnh cao…")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LAMA_ONNX_URL, dest)
    return dest


def _get_lama_onnx(log: Optional[LogFn] = None):
    global _LAMA_ONNX_SESS
    if _LAMA_ONNX_SESS is not None:
        return _LAMA_ONNX_SESS
    import onnxruntime as ort

    path = _ensure_lama_onnx(log)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    so.intra_op_num_threads = min(8, os.cpu_count() or 4)
    _LAMA_ONNX_SESS = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return _LAMA_ONNX_SESS


def _lama_onnx_inpaint(rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    import cv2

    sess = _get_lama_onnx()
    h, w = rgb.shape[:2]
    m = mask_u8
    binm = (m > 0).astype(np.uint8) if m.max() <= 1 else (m > 16).astype(np.uint8)
    if binm.sum() == 0:
        return rgb.copy()

    img_resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(binm, (512, 512), interpolation=cv2.INTER_NEAREST)

    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))[None, ...]
    mask_tensor = (mask_resized > 0).astype(np.float32)[None, None, ...]

    inps = {
        sess.get_inputs()[0].name: img_tensor,
        sess.get_inputs()[1].name: mask_tensor,
    }
    out = sess.run(None, inps)[0]
    out_img = np.clip(out[0], 0, 1) if out.max() <= 1.0 else np.clip(out[0] / 255.0, 0, 1)
    out_img = np.transpose(out_img, (1, 2, 0))
    out_u8 = (out_img * 255.0).astype(np.uint8)
    out_full = cv2.resize(out_u8, (w, h), interpolation=cv2.INTER_CUBIC)

    # Giữ nguyên 100% pixel gốc không bị mask
    mask_3c = (binm > 0)[..., None]
    return np.where(mask_3c, out_full, rgb)


MIGAN_URL = "https://huggingface.co/lxfater/inpaint-web/resolve/main/migan.onnx"
MIGAN_SHA = "bb7189b2523b8485d9dd6baa2e7e8bccce4493760daa33ffcd432f667945bf62"
_MIGAN_SESS = None


def _migan_path() -> Path:
    return get_model_path("migan.onnx")


def migan_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_migan(log: Optional[LogFn] = None) -> Path:
    import urllib.request

    dest = _migan_path()
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    if log:
        log("Đang tải model MIGAN (~29MB) lần đầu…")
    urllib.request.urlretrieve(MIGAN_URL, dest)
    if log:
        log(f"→ Đã lưu {dest}")
    return dest


def _get_migan(log: Optional[LogFn] = None):
    global _MIGAN_SESS
    if _MIGAN_SESS is not None:
        return _MIGAN_SESS
    import onnxruntime as ort

    path = _ensure_migan(log)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    avail = []
    try:
        avail = list(ort.get_available_providers())
    except Exception:
        avail = ["CPUExecutionProvider"]
    providers = []
    for name in (
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ):
        if name in avail:
            providers.append(name)
    if not providers:
        providers = ["CPUExecutionProvider"]
    if log:
        log(f"AI engine: {providers[0]}")
    _MIGAN_SESS = ort.InferenceSession(str(path), so, providers=providers)
    return _MIGAN_SESS


def _migan_inpaint(rgb: np.ndarray, mask_u8: np.ndarray, contexts=(3.0, 2.0)) -> np.ndarray:
    out = rgb
    for ctx in contexts:
        out = _migan_once(out, mask_u8, float(ctx))
    return out


def _migan_once(rgb: np.ndarray, mask_u8: np.ndarray, context: float) -> np.ndarray:
    from PIL import Image

    sess = _get_migan()
    h, w = rgb.shape[:2]
    m = mask_u8
    if m.max() <= 1:
        binm = (m > 0).astype(np.uint8)
    else:
        binm = (m > 16).astype(np.uint8)
    ys, xs = np.where(binm > 0)
    if ys.size == 0:
        return rgb.copy()
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    side = int(min(max(512, max(bw, bh) * context), w, h))
    cx, cy = x0 + bw / 2, y0 + bh / 2
    left = int(max(0, min(w - side, round(cx - side / 2))))
    top = int(max(0, min(h - side, round(cy - side / 2))))

    crop = rgb[top : top + side, left : left + side]
    cm = binm[top : top + side, left : left + side]
    img512 = np.array(Image.fromarray(crop).resize((512, 512), Image.Resampling.BILINEAR))
    m512 = np.array(
        Image.fromarray((cm * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST)
    )
    keep = (m512 <= 64).astype(np.float32)
    rgb01 = img512.astype(np.float32) / 255.0 * 2.0 - 1.0
    inp = np.zeros((1, 4, 512, 512), dtype=np.float32)
    inp[0, 0] = keep - 0.5
    for c in range(3):
        inp[0, c + 1] = rgb01[:, :, c] * keep

    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0]
    pred = out[0]
    if pred.shape[0] == 3:
        pr = np.transpose(pred, (1, 2, 0))
    else:
        pr = pred
    pr = np.clip(0.5 * pr + 0.5, 0, 1)
    pred_u8 = (pr * 255).astype(np.uint8)
    pred_u8 = np.array(Image.fromarray(pred_u8).resize((side, side), Image.Resampling.BILINEAR))

    feather = np.array(
        Image.fromarray((cm * 255).astype(np.uint8)).resize((side, side), Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0
    try:
        import cv2
        k = max(3, side // 80)
        if k % 2 == 0:
            k += 1
        feather = cv2.GaussianBlur(feather, (k, k), 0)
    except Exception:
        pass
    alpha = feather[..., None]
    blended = pred_u8.astype(np.float32) * alpha + crop.astype(np.float32) * (1 - alpha)
    result = rgb.copy()
    result[top : top + side, left : left + side] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _tiled_migan_inpaint(rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Tự động chia ô (tiled sliding window) để AI MIGAN inpaint toàn bộ các vùng khuyết lớn/chéo."""
    h, w = rgb.shape[:2]
    m = mask_u8
    binm = (m > 0).astype(np.uint8) if m.max() <= 1 else (m > 16).astype(np.uint8)
    ys, xs = np.where(binm > 0)
    if ys.size == 0:
        return rgb.copy()

    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)

    # Nếu vùng mask nhỏ và ảnh không quá to, chạy MIGAN tiêu chuẩn
    if max(bw, bh) <= 480 and max(w, h) <= 900:
        return _migan_inpaint(rgb, mask_u8, contexts=(3.0, 2.0))

    # Chạy Tiled Inpainting với sliding window 512x512
    tile_size = 512
    stride = 360  # Overlap 152px để không lộ viền
    out = rgb.copy()

    for y0 in range(0, max(1, h - tile_size + stride), stride):
        for x0 in range(0, max(1, w - tile_size + stride), stride):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            y0_adj = max(0, y1 - tile_size)
            x0_adj = max(0, x1 - tile_size)

            tile_mask = binm[y0_adj:y1, x0_adj:x1]
            if tile_mask.sum() == 0:
                continue

            tile_rgb = out[y0_adj:y1, x0_adj:x1]
            inp_tile = _migan_once(tile_rgb, tile_mask, context=1.6)
            out[y0_adj:y1, x0_adj:x1] = inp_tile

    return out


def extract_text_watermark_mask(rgb: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """Tự động bóc tách toàn bộ các hàng watermark chéo lặp lại dày đặc mà không xóa nhầm chi tiết ảnh gốc."""
    import cv2
    h, w = rgb.shape[:2]

    # 1. Bóc tách nét chữ mảnh & tương phản cục bộ qua Top-Hat + Gradient
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k_size = int(max(5, min(h, w) * 0.015))
    if k_size % 2 == 0:
        k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    comp = cv2.addWeighted(tophat, 0.7, lap, 0.3, 0)

    # Ngưỡng thích ứng bám theo độ nhạy người dùng
    pct = max(78.0, min(94.0, 90.0 - (float(sensitivity) * 8.0)))
    thr = max(6.0, float(np.percentile(comp, pct)))
    _, stroke_mask = cv2.threshold(comp, thr, 255, cv2.THRESH_BINARY)

    # 2. Tạo khung bao phủ toàn bộ các dải chéo song song lặp lại toàn màn hình
    stripe_mask = np.zeros((h, w), dtype=np.uint8)
    spacing = int(min(w, h) * 0.22)
    thick = int(max(20, min(w, h) * 0.09))
    for offset in range(-h * 2, w * 2, spacing):
        cv2.line(stripe_mask, (offset, h + 40), (offset + int(w * 1.3), -40), 255, thickness=thick)

    # 3. Kết hợp: Bám chặt nét chữ trong các dải chéo để giữ nguyên 100% chi tiết cây cối/cảnh vật
    combined = cv2.bitwise_and(stroke_mask, stripe_mask)
    if combined.sum() < 200:
        combined = stripe_mask

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(combined, dilate_k, iterations=1)


def inpaint_mask(rgb: np.ndarray, mask: np.ndarray, use_ai: bool = True, prefer_gpu: bool = False) -> tuple[np.ndarray, dict]:
    """Inpaint theo một mặt nạ nhị phân bất kỳ (từ Brush hoặc Auto Detector)."""
    if mask is None or int(mask.sum()) == 0:
        return rgb.copy(), {"applied": False, "reason": "EMPTY_MASK"}

    m_u8 = (mask > 0).astype(np.uint8) * 255
    if use_ai:
        if prefer_gpu:
            if migan_available():
                try:
                    out = _tiled_migan_inpaint(rgb, m_u8)
                    return out, {"applied": True, "kind": "custom_mask", "engine": "migan_gpu"}
                except Exception:
                    pass
            if lama_onnx_available():
                try:
                    out = _lama_onnx_inpaint(rgb, m_u8)
                    return out, {"applied": True, "kind": "custom_mask", "engine": "lama_onnx"}
                except Exception:
                    pass
        else:
            if lama_onnx_available():
                try:
                    out = _lama_onnx_inpaint(rgb, m_u8)
                    return out, {"applied": True, "kind": "custom_mask", "engine": "lama_onnx"}
                except Exception:
                    pass
            if migan_available():
                try:
                    out = _tiled_migan_inpaint(rgb, m_u8)
                    return out, {"applied": True, "kind": "custom_mask", "engine": "migan_tiled"}
                except Exception:
                    pass
        if lama_available():
            out = _lama_inpaint(rgb, m_u8)
            return out, {"applied": True, "kind": "custom_mask", "engine": "lama"}

    try:
        import cv2
        out = cv2.inpaint(rgb, m_u8, 3, cv2.INPAINT_TELEA)
        return out, {"applied": True, "kind": "custom_mask", "engine": "opencv_telea"}
    except Exception:
        out = _simple_inpaint(rgb, (m_u8 > 0).astype(np.uint8), radius=4)
        return out, {"applied": True, "kind": "custom_mask", "engine": "numpy_inpaint"}


def extract_horizontal_text_mask(rgb: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """Tự động bóc tách chính xác từng nét chữ ngang ở giữa hoặc chân ảnh mà không làm mất chi tiết bờ biển/phong cảnh phía sau."""
    import cv2
    h, w = rgb.shape[:2]

    # 1. Bóc tách nét chữ tương phản qua Dual-Hat & Laplacian
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k_size = int(max(5, min(h, w) * 0.015))
    if k_size % 2 == 0:
        k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    dual = cv2.max(tophat, blackhat)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    comp = cv2.addWeighted(dual, 0.7, lap, 0.3, 0)

    # 2. Vùng giới hạn chữ ngang (y từ 35% đến 68%)
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(h * 0.35) : int(h * 0.68), int(w * 0.02) : int(w * 0.98)] = 255
    comp_roi = cv2.bitwise_and(comp, roi)

    pct = max(75.0, min(92.0, 88.0 - (float(sensitivity) * 8.0)))
    thr = max(5.0, float(np.percentile(comp_roi[comp_roi > 0] if comp_roi.sum() > 0 else comp_roi, pct)))
    _, stroke_mask = cv2.threshold(comp_roi, thr, 255, cv2.THRESH_BINARY)

    # Morphological close để liền mạch nét chữ nhưng không lan ra phong cảnh xung quanh
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, k_close)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask_clean, dilate_k, iterations=1)


def extract_smart_watermark_mask(rgb: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """Universal AI Multi-Watermark Detector:
    Tự động dò tìm và bóc tách mọi loại watermark (chữ sáng, chữ tối, dải chéo, dải ngang, logo 4 góc).
    """
    import cv2
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Dual-Hat Filter (Bắt cả chữ sáng trên nền tối và chữ tối trên nền sáng)
    k_size = int(max(5, min(h, w) * 0.015))
    if k_size % 2 == 0:
        k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    dual_hat = cv2.max(tophat, blackhat)

    # 2. Gradient / Edge Response
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    comp = cv2.addWeighted(dual_hat, 0.7, lap, 0.3, 0)

    # Ngưỡng thích ứng
    pct = max(76.0, min(95.0, 89.0 - (float(sensitivity) * 8.0)))
    thr = max(5.0, float(np.percentile(comp, pct)))
    _, stroke_mask = cv2.threshold(comp, thr, 255, cv2.THRESH_BINARY)

    # 3. Lọc nhiễu hạt nhỏ
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(stroke_mask, connectivity=8)
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if 4 <= area <= (w * h * 0.15) and (bw >= 3 or bh >= 3):
            clean_mask[output == i] = 255

    # 4. Kiểm tra xem có dải chéo hoặc dải ngang không
    stripe_mask = np.zeros((h, w), dtype=np.uint8)
    spacing = int(min(w, h) * 0.22)
    thick = int(max(20, min(w, h) * 0.09))
    for offset in range(-h * 2, w * 2, spacing):
        cv2.line(stripe_mask, (offset, h + 40), (offset + int(w * 1.3), -40), 255, thickness=thick)

    combined = cv2.bitwise_and(clean_mask, stripe_mask)
    if combined.sum() > 500:
        clean_mask = combined

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(clean_mask, dilate_k, iterations=1)


def remove_stock_text_watermark(rgb: np.ndarray, use_ai: bool = True, sensitivity: float = 0.5, prefer_gpu: bool = False) -> tuple[np.ndarray, dict]:
    """Tự động nhận diện và xóa watermark chữ chéo / stock photo watermark toàn màn hình."""
    mask = extract_text_watermark_mask(rgb, sensitivity=sensitivity)
    out, info = inpaint_mask(rgb, mask, use_ai=use_ai, prefer_gpu=prefer_gpu)
    info["kind"] = "stock_text_diagonal"
    info["masked_pixels"] = int(mask.sum() // 255)
    return out, info


def remove_horizontal_text_watermark(rgb: np.ndarray, use_ai: bool = True, sensitivity: float = 0.5, prefer_gpu: bool = False) -> tuple[np.ndarray, dict]:
    """Tự động nhận diện và xóa watermark chữ ngang / subtitle ở giữa hoặc chân ảnh."""
    mask = extract_horizontal_text_mask(rgb, sensitivity=sensitivity)
    out, info = inpaint_mask(rgb, mask, use_ai=use_ai, prefer_gpu=prefer_gpu)
    info["kind"] = "horizontal_text_banner"
    info["masked_pixels"] = int(mask.sum() // 255)
    return out, info


def remove_smart_auto_watermark(rgb: np.ndarray, use_ai: bool = True, sensitivity: float = 0.5, prefer_gpu: bool = False) -> tuple[np.ndarray, dict]:
    """Tự động thông minh an toàn: Dò tìm logo 4 góc và biểu tượng sparkle AI (Gemini, Veo, TikTok, CapCut).
    Tuyệt đối không xóa lan vào chủ thể chính, đồ ăn hay khuôn mặt ở giữa ảnh nếu không có watermark rõ ràng."""
    out_c, info_c = remove_watermark(rgb, prefer_corner=True)
    if info_c.get("applied"):
        b = info_c.get("box") or info_c.get("pos")
        if b and use_ai:
            box_dict = {
                "x": int(b.get("x", 0)),
                "y": int(b.get("y", 0)),
                "w": int(b.get("w", b.get("size", 40))),
                "h": int(b.get("h", b.get("size", 40))),
            }
            return inpaint_box(rgb, box_dict, use_ai=True, prefer_gpu=prefer_gpu)
        return out_c, info_c

    # Không tìm thấy watermark góc rõ ràng -> Trả về ảnh gốc an toàn, không xóa bừa bãi
    return rgb.copy(), {
        "applied": False,
        "reason": "Không phát hiện watermark góc hoặc sparkle AI rõ ràng. Vui lòng dùng chế độ '🎯 Thủ công (Bút vẽ & Khoanh ô)' để khoanh vùng chữ cần xóa.",
        "kind": "none",
        "engine": "none",
    }


def ai_inpaint(rgb: np.ndarray, mask_u8: np.ndarray, log: Optional[LogFn] = None, passes: int = 2, prefer_gpu: bool = False) -> tuple:
    if prefer_gpu and migan_available():
        try:
            return _tiled_migan_inpaint(rgb, mask_u8), "migan_gpu"
        except Exception as e:
            if log:
                log(f"MIGAN GPU lỗi, thử cách khác: {e}")
    if lama_onnx_available():
        try:
            return _lama_onnx_inpaint(rgb, mask_u8), "lama_onnx"
        except Exception as e:
            if log:
                log(f"LaMa ONNX lỗi: {e}")
    if migan_available():
        try:
            return _tiled_migan_inpaint(rgb, mask_u8), "migan"
        except Exception as e:
            if log:
                log(f"MIGAN lỗi, thử cách khác: {e}")
    if lama_available():
        return _lama_inpaint(rgb, mask_u8), "lama"
    raise RuntimeError(
        "Chưa có AI inpaint. Cài onnxruntime."
    )


def _which_ffmpeg() -> tuple[str, str]:
    return get_ffmpeg_path(), get_ffprobe_path()


_CACHED_BEST_ENCODER: tuple[str, list[str]] | None = None

def _detect_best_encoder(ffmpeg: str) -> tuple[str, list[str]]:
    """Tự động kiểm tra và kích hoạt GPU hardware encoder (có cache để chạy siêu tốc)."""
    global _CACHED_BEST_ENCODER
    if _CACHED_BEST_ENCODER is not None:
        return _CACHED_BEST_ENCODER

    encoders = [
        ("h264_nvenc", ["-preset", "p4", "-cq", "19"]),
        ("h264_qsv", ["-preset", "medium", "-global_quality", "20"]),
        ("h264_amf", ["-quality", "speed", "-rc", "cbr"]),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    for enc, args in encoders:
        try:
            cmd = [
                ffmpeg, "-hide_banner", "-v", "error",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                "-c:v", enc, "-f", "null", "-",
            ]
            proc = subprocess.run(cmd, capture_output=True, creationflags=flags, timeout=1.5)
            if proc.returncode == 0:
                _CACHED_BEST_ENCODER = (enc, args)
                return _CACHED_BEST_ENCODER
        except Exception:
            pass
    _CACHED_BEST_ENCODER = ("libx264", ["-crf", "16", "-preset", "medium"])
    return _CACHED_BEST_ENCODER


def get_hardware_acceleration_info() -> dict:
    """Kiểm tra và trả về thông tin tăng tốc phần cứng (AI Providers & Video Encoders)."""
    ffmpeg, _ = _which_ffmpeg()
    enc, _ = _detect_best_encoder(ffmpeg)
    enc_desc = {
        "h264_nvenc": "NVIDIA NVENC",
        "h264_qsv": "Intel QuickSync",
        "h264_amf": "AMD AMF",
        "libx264": "CPU (libx264)",
    }.get(enc, enc)

    ai_provider = "CPU"
    try:
        import onnxruntime as ort
        avail = list(ort.get_available_providers())
        if "DmlExecutionProvider" in avail:
            ai_provider = "DirectML (GPU)"
        elif "CUDAExecutionProvider" in avail:
            ai_provider = "NVIDIA CUDA (GPU)"
        elif "ROCMExecutionProvider" in avail:
            ai_provider = "AMD ROCm"
    except Exception:
        ai_provider = "CPU (Fallback)"

    return {
        "encoder": enc,
        "encoder_desc": enc_desc,
        "ai_provider": ai_provider,
        "is_gpu_accelerated": enc != "libx264" or "GPU" in ai_provider or "DirectML" in ai_provider,
    }


def process_video(
    src: Path,
    dst: Path,
    log: Optional[LogFn] = None,
    max_seconds: float = 300.0,
    box: Optional[dict] = None,
    use_ai: bool = False,
    progress_fn: Optional[Callable[[int, int, float], None]] = None,
    cancel_event: Optional[object] = None,
    mode: str = "auto",
    custom_mask: Optional[np.ndarray] = None,
) -> dict:
    """Xóa watermark video trực tiếp trong RAM qua FFmpeg Pipe & GPU Acceleration."""
    def _log(m: str) -> None:
        if log:
            log(m)

    src, dst = Path(src).resolve(), Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = _which_ffmpeg()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    try:
        raw = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(src),
            ],
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        meta = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        raise RuntimeError(f"Không đọc được video (cần FFmpeg/ffprobe): {e}") from e

    width, height = 0, 0
    duration = 0.0
    fps = 30.0
    has_audio = False
    total_frames = 0

    for st in meta.get("streams") or []:
        if st.get("codec_type") == "video":
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
            fr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
            try:
                a, b = fr.split("/")
                fps = float(a) / max(float(b), 1e-9)
            except Exception:
                fps = 30.0
            nb_f = st.get("nb_frames")
            if nb_f and str(nb_f).isdigit():
                total_frames = int(nb_f)
        if st.get("codec_type") == "audio":
            has_audio = True

    try:
        duration = float((meta.get("format") or {}).get("duration") or 0)
    except Exception:
        duration = 0.0

    if width <= 0 or height <= 0:
        raise ValueError("Không xác định được kích thước khung hình video.")

    if duration > max_seconds:
        raise ValueError(
            f"Video dài {duration:.0f}s — giới hạn {max_seconds:.0f}s/lần."
        )

    if total_frames == 0 and duration > 0:
        total_frames = int(duration * fps)

    enc, enc_args = _detect_best_encoder(ffmpeg)
    enc_desc = {
        "h264_nvenc": "NVIDIA GPU (NVENC)",
        "h264_qsv": "Intel QuickSync (QSV)",
        "h264_amf": "AMD GPU (AMF)",
        "libx264": "CPU (libx264)",
    }.get(enc, enc)
    _log(f"Bắt đầu pipeline RAM ({width}x{height} @ {fps:.2f}fps) · Encoder: {enc_desc}")

    read_cmd = [
        ffmpeg,
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(src),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    write_cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.5f}",
        "-i",
        "pipe:0",
    ]

    if has_audio:
        write_cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]

    write_cmd += (
        ["-c:v", enc]
        + enc_args
        + ["-pix_fmt", "yuv420p", "-shortest", str(dst)]
    )

    reader = subprocess.Popen(
        read_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
    )
    writer = subprocess.Popen(
        write_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags,
    )

    frame_bytes = width * height * 3
    applied = 0
    frame_idx = 0
    import time
    start_time = time.time()
    last_prog_time = start_time

    try:
        while True:
            if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                _log("! Đã nhận lệnh HỦY từ người dùng.")
                try:
                    reader.kill()
                except Exception:
                    pass
                try:
                    writer.kill()
                except Exception:
                    pass
                if dst.is_file():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                return {
                    "applied": False,
                    "cancelled": True,
                    "frames": frame_idx,
                    "marked_frames": applied,
                    "dst": str(dst),
                    "src": str(src),
                }

            raw_frame = reader.stdout.read(frame_bytes)
            if not raw_frame or len(raw_frame) < frame_bytes:
                break

            frame_idx += 1
            rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

            if custom_mask is not None and int(custom_mask.sum()) > 0:
                out, info = inpaint_mask(rgb, custom_mask, use_ai=use_ai, prefer_gpu=True)
            elif box:
                out, info = inpaint_box(rgb, box, use_ai=use_ai, ai_passes=1 if use_ai else 2, prefer_gpu=True)
            elif mode == "stock_text":
                out, info = remove_stock_text_watermark(rgb, use_ai=use_ai, prefer_gpu=True)
            elif mode == "horizontal_text":
                out, info = remove_horizontal_text_watermark(rgb, use_ai=use_ai, prefer_gpu=True)
            else:
                out, info = remove_smart_auto_watermark(rgb, use_ai=use_ai, prefer_gpu=True)

            if info.get("applied"):
                applied += 1

            writer.stdin.write(out.tobytes())

            now = time.time()
            if progress_fn and (now - last_prog_time >= 0.1 or frame_idx == total_frames):
                elapsed = max(0.001, now - start_time)
                cur_fps = frame_idx / elapsed
                progress_fn(frame_idx, total_frames, cur_fps)
                last_prog_time = now

            if frame_idx == 1 or frame_idx % 30 == 0 or (total_frames and frame_idx == total_frames):
                tot = f"/{total_frames}" if total_frames else ""
                elapsed = max(0.001, now - start_time)
                cur_fps = frame_idx / elapsed
                _log(f"  xử lý frame {frame_idx}{tot} ({cur_fps:.1f} fps - đã xóa: {applied})")

        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()

        if writer.returncode != 0:
            err = (writer.stderr.read() or b"").decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"Lỗi encode video (code {writer.returncode}):\n{err}")

        _log(f"✓ Video → {dst.name} (tổng {frame_idx} frame, đã xóa mark: {applied})")
        return {
            "applied": applied > 0,
            "frames": frame_idx,
            "marked_frames": applied,
            "dst": str(dst),
            "src": str(src),
        }
    except Exception:
        try:
            reader.kill()
        except Exception:
            pass
        try:
            writer.kill()
        except Exception:
            pass
        raise


def extract_preview_frame(src: Path, dst: Path, time_sec: float = 0.3) -> Path:
    ffmpeg, _ = _which_ffmpeg()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-ss", str(max(0.0, time_sec)),
        "-i", str(src),
        "-frames:v", "1",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    if proc.returncode != 0 or not dst.is_file():
        cmd = [ffmpeg, "-hide_banner", "-y", "-i", str(src), "-frames:v", "1", str(dst)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    if not dst.is_file():
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"Không lấy được frame xem trước.\n{err}")
    return dst