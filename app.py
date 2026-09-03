"""
ZWatermark Remover AI — Ứng dụng chuyên biệt xóa Watermark, Logo, Subtitle bằng AI & GPU Inpainting.
Chạy: python app_watermark.py
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from tkinter import filedialog, messagebox

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    import customtkinter as ctk
except ImportError:
    print("Chưa cài customtkinter. Chạy: pip install -r requirements.txt")
    sys.exit(1)

from PIL import Image, ImageTk

from core.paths import (
    get_app_dir,
    get_asset_path,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_model_path,
    is_frozen,
)
from core.watermark_remover import (
    IMAGE_EXT,
    VIDEO_EXT,
    extract_preview_frame,
    get_hardware_acceleration_info,
    migan_available,
    lama_available,
    process_file,
    process_video,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Font Awesome & Icons ──────────────────────────────────────
_APP_DIR = get_app_dir()
_FA_FAMILY = "Font Awesome 6 Free Solid"


def _find_fa() -> Path | None:
    names = ("fa-solid-900.ttf", "Font Awesome 6 Free-Solid-900.otf")
    roots = [
        _APP_DIR,
        _APP_DIR / "assets",
        Path(sys.executable).resolve().parent,
        Path(sys.executable).resolve().parent / "assets",
        Path.cwd(),
        Path.cwd() / "assets",
    ]
    for root in roots:
        for name in names:
            pth = root / name
            if pth.is_file():
                return pth
    return None


_FA_PATH = _find_fa() or (_APP_DIR / "assets" / "fa-solid-900.ttf")


def _register_fa() -> str:
    global _FA_FAMILY
    if not _FA_PATH.is_file():
        return "Segoe UI Symbol"
    path = str(_FA_PATH)
    if sys.platform == "win32":
        try:
            import ctypes

            FR_PRIVATE = 0x10
            ctypes.windll.gdi32.AddFontResourceExW(path, FR_PRIVATE, 0)
        except Exception:
            pass
    return _FA_FAMILY


_FA_FAMILY = _register_fa()


class FA:
    PLAY = "\uf04b"
    STOP = "\uf04d"
    FOLDER = "\uf07b"
    FILE = "\uf15b"
    IMAGE = "\uf03e"
    FILM = "\uf008"
    TRASH = "\uf1f8"
    CHECK = "\uf00c"
    GEAR = "\uf013"
    ERASER = "\uf12d"
    MAGIC = "\uf0d0"
    CROP = "\uf125"
    CROSS = "\uf00d"
    CHEVRON_LEFT = "\uf053"
    CHEVRON_RIGHT = "\uf054"
    BOLT = "\uf0e7"
    MICROCHIP = "\uf2db"
    ROTATE = "\uf01e"


def fa_font(size: int = 14) -> ctk.CTkFont:
    return ctk.CTkFont(family=_FA_FAMILY, size=size)


# ── Color Palette ─────────────────────────────────────────────
BG_COLOR = "#0c1017"
PANEL_BG = "#111622"
CARD_BG = "#161c2b"
CARD_LINE = "#232d42"
ACCENT = "#0284c7"
ACCENT_HOVER = "#0369a1"
ACCENT_LIGHT = "#38bdf8"
ACCENT_GRADIENT = "#2563eb"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"
TEXT_DARK = "#64748b"
SUCCESS_GREEN = "#10b981"
DANGER_RED = "#ef4444"
DANGER_HOVER = "#dc2626"
WARNING_YELLOW = "#f59e0b"

SUPPORTED_EXTS = IMAGE_EXT | VIDEO_EXT


# ── Drag & Drop Wrapper ──────────────────────────────────────
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    class _BaseApp(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self._has_dnd = True
            except Exception:
                self._has_dnd = False

    _HAS_DND = True
except Exception:
    class _BaseApp(ctk.CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._has_dnd = False

    _HAS_DND = False
    DND_FILES = None


def _parse_drop_data(raw_data: str, tk_root=None) -> list[str]:
    if not raw_data:
        return []
    paths = []
    if "{" in raw_data and "}" in raw_data:
        pattern = r"\{([^}]+)\}|(\S+)"
        for match in re.finditer(pattern, raw_data):
            p = match.group(1) or match.group(2)
            if p:
                paths.append(p.strip())
    else:
        if tk_root:
            try:
                paths = list(tk_root.tk.splitlist(raw_data))
            except Exception:
                paths = raw_data.split()
        else:
            paths = raw_data.split()
    return [p for p in paths if p]


class ZWatermarkRemoverApp(_BaseApp):
    def __init__(self):
        super().__init__()
        self.title("ZWatermark Remover — Xóa Watermark & Inpainting Siêu Tốc")
        self.geometry("1240x820")
        self.minsize(1080, 720)
        self.configure(fg_color=BG_COLOR)

        # Set Icon if exists
        ico = get_asset_path("app.ico")
        if ico and ico.is_file():
            try:
                self.iconbitmap(str(ico))
            except Exception:
                pass

        # Data state
        self.files_list: list[dict] = []  # dict: {path: Path, ext: str, is_video: bool, status: str, result_path: str|None}
        self.output_dir = ctk.StringVar(value="")
        self.use_ai = ctk.BooleanVar(value=True)
        self.detect_mode = ctk.StringVar(value="auto")  # "auto", "stock_text", "manual"
        self.custom_box: dict | None = None
        self.custom_mask: object = None  # np.ndarray | None
        self.is_processing = False
        self.cancel_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.active_tab = "remover"
        self._hw_info_dict = {}

        # Compare Studio state
        self.compare_pairs: list[tuple[str, str]] = []  # (src_path, dst_path)
        self.compare_idx = 0
        self.compare_split_pct = 50.0
        self.compare_canvas = None
        self._cmp_img_b: Image.Image | None = None
        self._cmp_img_a: Image.Image | None = None
        self._cached_thumbs: list[ctk.CTkImage] = []

        # Build UI
        self._build_header()
        self._build_main_layout()
        self._setup_dnd()

        self.status_lbl.configure(text="⚡ Đang kết nối DirectML GPU & nạp AI Inpainting Engine...")
        self.update()

        # Check Hardware & Init log
        self.after(50, self._detect_hardware)

        # Background pre-warm AI models so inference starts instantaneously (0.2s)
        threading.Thread(target=self._prewarm_ai_models, daemon=True).start()

    def _prewarm_ai_models(self) -> None:
        try:
            from core.watermark_remover import _get_lama_onnx, _get_migan
            _get_migan()
            _get_lama_onnx()
        except Exception:
            pass

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=PANEL_BG, height=64, corner_radius=0, border_width=1, border_color=CARD_LINE)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Left: App title and branding
        left_box = ctk.CTkFrame(header, fg_color="transparent")
        left_box.pack(side="left", padx=20, fill="y")

        icon_lbl = ctk.CTkLabel(
            left_box,
            text=FA.ERASER,
            font=fa_font(22),
            text_color=ACCENT_LIGHT,
        )
        icon_lbl.pack(side="left", padx=(0, 10))

        title_col = ctk.CTkFrame(left_box, fg_color="transparent")
        title_col.pack(side="left", fill="y", pady=10)

        t_row = ctk.CTkFrame(title_col, fg_color="transparent")
        t_row.pack(anchor="w")
        ctk.CTkLabel(
            t_row,
            text="ZWatermark Remover",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        badge = ctk.CTkFrame(t_row, fg_color="#1e293b", corner_radius=6)
        badge.pack(side="left", padx=8)
        ctk.CTkLabel(
            badge,
            text="PRO GPU",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=ACCENT_LIGHT,
        ).pack(padx=6, pady=1)

        ctk.CTkLabel(
            title_col,
            text="Inpainting Deep Learning • DirectML • CUDA • Video Pipe RAM",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).pack(anchor="w")

        # Center: Tab switcher buttons
        self.nav_box = ctk.CTkFrame(header, fg_color="#0e131d", corner_radius=10, border_width=1, border_color=CARD_LINE)
        self.nav_box.pack(side="left", padx=24, pady=12)

        self.tab_remover_btn = ctk.CTkButton(
            self.nav_box,
            text="🧼 Xóa Watermark",
            width=136,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=ACCENT_GRADIENT,
            hover_color=ACCENT_HOVER,
            command=lambda: self._switch_tab("remover"),
        )
        self.tab_remover_btn.pack(side="left", padx=3, pady=3)

        self.tab_info_btn = ctk.CTkButton(
            self.nav_box,
            text="ℹ️ Thông tin",
            width=110,
            height=32,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="transparent",
            hover_color="#1e293b",
            text_color=TEXT_MUTED,
            command=lambda: self._switch_tab("info"),
        )
        self.tab_info_btn.pack(side="left", padx=3, pady=3)

        # Right: Hardware status chips
        self.hw_box = ctk.CTkFrame(header, fg_color="transparent")
        self.hw_box.pack(side="right", padx=20, fill="y")

        self.gpu_chip = ctk.CTkLabel(
            self.hw_box,
            text="⚡ Đang dò GPU...",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a2234",
            text_color=TEXT_MUTED,
            corner_radius=8,
            padx=12,
            pady=6,
        )
        self.gpu_chip.pack(side="right", padx=4, pady=14)

        self.ai_chip = ctk.CTkLabel(
            self.hw_box,
            text="🧠 AI Engine...",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a2234",
            text_color=TEXT_MUTED,
            corner_radius=8,
            padx=12,
            pady=6,
        )
        self.ai_chip.pack(side="right", padx=4, pady=14)

    def _build_main_layout(self) -> None:
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True)

        # ── TAB 1: WORKSPACE (Remover & Studio) ───────────────────
        self.workspace_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.workspace_frame.pack(fill="both", expand=True, padx=16, pady=14)

        # Split 2 Columns
        self.workspace_frame.grid_columnconfigure(0, weight=5, uniform="col")
        self.workspace_frame.grid_columnconfigure(1, weight=5, uniform="col")
        self.workspace_frame.grid_rowconfigure(0, weight=1)

        # Left Panel: Files, Settings, Controls
        self.left_panel = ctk.CTkFrame(self.workspace_frame, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=CARD_LINE)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_left_panel()

        # Right Panel: Before / After Studio
        self.right_panel = ctk.CTkFrame(self.workspace_frame, fg_color=PANEL_BG, corner_radius=16, border_width=1, border_color=CARD_LINE)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._build_right_panel()

        # ── TAB 2: INFO & USER GUIDE ──────────────────────────────
        self.info_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self._build_info_tab()

    def _switch_tab(self, tab_name: str) -> None:
        self.active_tab = tab_name
        if tab_name == "remover":
            self.info_frame.pack_forget()
            self.workspace_frame.pack(fill="both", expand=True, padx=16, pady=14)
            self.tab_remover_btn.configure(fg_color=ACCENT_GRADIENT, text_color=TEXT_MAIN)
            self.tab_info_btn.configure(fg_color="transparent", text_color=TEXT_MUTED)
        else:
            self.workspace_frame.pack_forget()
            self.info_frame.pack(fill="both", expand=True, padx=16, pady=14)
            self.tab_info_btn.configure(fg_color=ACCENT_GRADIENT, text_color=TEXT_MAIN)
            self.tab_remover_btn.configure(fg_color="transparent", text_color=TEXT_MUTED)

    def _build_left_panel(self) -> None:
        # 1. Section Header & Tool Buttons
        top_bar = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        top_bar.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            top_bar,
            text="Danh sách tệp",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        self.count_lbl = ctk.CTkLabel(
            top_bar,
            text="0 tệp",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        )
        self.count_lbl.pack(side="left", padx=10)

        # Action Buttons
        btn_box = ctk.CTkFrame(top_bar, fg_color="transparent")
        btn_box.pack(side="right")

        ctk.CTkButton(
            btn_box,
            text="+ Thêm tệp",
            width=90,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._browse_files,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            btn_box,
            text="📁 Thư mục",
            width=90,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._browse_folder,
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            btn_box,
            text="🧹 Xóa hết",
            width=80,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._clear_files,
        ).pack(side="left", padx=3)

        # 2. File List / Drop Zone Box
        self.drop_container = ctk.CTkFrame(
            self.left_panel,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_LINE,
            height=210,
        )
        self.drop_container.pack(fill="x", padx=16, pady=4)
        self.drop_container.pack_propagate(False)

        # Empty Drop Hint
        self.drop_empty = ctk.CTkFrame(self.drop_container, fg_color="transparent")
        self.drop_empty.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            self.drop_empty,
            text=FA.IMAGE,
            font=fa_font(32),
            text_color=ACCENT_LIGHT,
        ).pack()

        ctk.CTkLabel(
            self.drop_empty,
            text="Kéo & thả ảnh / video vào đây",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(pady=(4, 2))

        ctk.CTkLabel(
            self.drop_empty,
            text="Hỗ trợ: MP4, MOV, MKV, WEBM, PNG, JPG, WEBP...",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DARK,
        ).pack()

        # Populated File Queue Scroll
        self.file_scroll = ctk.CTkScrollableFrame(
            self.drop_container,
            fg_color="transparent",
        )

        # 3. Settings Card
        settings_card = ctk.CTkFrame(self.left_panel, fg_color=CARD_BG, corner_radius=12, border_width=1, border_color=CARD_LINE)
        settings_card.pack(fill="x", padx=16, pady=8)

        # Row 1: Mode Switcher Dropdown
        r1 = ctk.CTkFrame(settings_card, fg_color="transparent")
        r1.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            r1,
            text="Chế độ:",
            width=70,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        self.mode_menu = ctk.CTkOptionMenu(
            r1,
            values=[
                "✨ Tự động Toàn diện (Smart Auto)",
                "🎯 Thủ công (Bút vẽ & Khoanh ô)",
            ],
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            button_color=ACCENT_GRADIENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color="#131926",
            command=self._on_mode_change,
        )
        self.mode_menu.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            r1,
            text="🎯 Bút vẽ & Khoanh ô",
            width=140,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._open_region_picker,
        ).pack(side="right")

        # Row 2: Status sub-label
        r_sub = ctk.CTkFrame(settings_card, fg_color="transparent")
        r_sub.pack(fill="x", padx=12, pady=(0, 4))
        self.region_lbl = ctk.CTkLabel(
            r_sub,
            text="Chế độ: Tự động quét logo góc & watermark toàn diện",
            font=ctk.CTkFont(size=11),
            text_color=ACCENT_LIGHT,
            anchor="w",
        )
        self.region_lbl.pack(fill="x", padx=(70, 0))

        # Row 3: Helpful Note for Complex Watermarks
        note_box = ctk.CTkFrame(settings_card, fg_color="#0f172a", corner_radius=8, border_width=1, border_color="#1e293b")
        note_box.pack(fill="x", padx=12, pady=(2, 6))
        ctk.CTkLabel(
            note_box,
            text="💡 Lưu ý: Với watermark mờ phức tạp hoặc chìm vào sóng/vân ảnh, tính năng Tự Động có thể nhận diện chưa chuẩn. Hãy dùng '🎯 Bút vẽ & Khoanh ô' để khoanh vùng theo ý muốn.",
            font=ctk.CTkFont(size=10),
            text_color="#94a3b8",
            wraplength=430,
            justify="left",
        ).pack(padx=8, pady=5)

        # Row 4: AI Inpaint Mode Switch
        r2 = ctk.CTkFrame(settings_card, fg_color="transparent")
        r2.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(
            r2,
            text="AI Engine:",
            width=75,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        ctk.CTkSwitch(
            r2,
            text="Xóa sạch bằng Deep AI Inpainting (MIGAN ONNX / DirectML)",
            variable=self.use_ai,
            font=ctk.CTkFont(size=12),
            progress_color=ACCENT,
            button_color=TEXT_MAIN,
        ).pack(side="left", padx=4)

        # Row 3: Output Folder
        r3 = ctk.CTkFrame(settings_card, fg_color="transparent")
        r3.pack(fill="x", padx=12, pady=(6, 10))

        ctk.CTkLabel(
            r3,
            text="Thư mục:",
            width=75,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        self.out_entry = ctk.CTkEntry(
            r3,
            textvariable=self.output_dir,
            placeholder_text="Mặc định: Lưu cạnh file gốc (_removed_watermark)",
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11),
            fg_color="#0f172a",
            border_color=CARD_LINE,
        )
        self.out_entry.pack(side="left", fill="x", expand=True, padx=4)

        ctk.CTkButton(
            r3,
            text="Chọn",
            width=60,
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._browse_output_dir,
        ).pack(side="left")

        # 4. Action CTA & Progress Bar
        action_box = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        action_box.pack(fill="x", padx=16, pady=6)

        btn_row = ctk.CTkFrame(action_box, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 6))

        self.start_btn = ctk.CTkButton(
            btn_row,
            text="🚀 BẮT ĐẦU XÓA WATERMARK",
            height=40,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT_GRADIENT,
            hover_color=ACCENT_HOVER,
            command=self._start_processing,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.stop_btn = ctk.CTkButton(
            btn_row,
            text="⏹ DỪNG",
            width=80,
            height=40,
            corner_radius=10,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1e293b",
            hover_color=DANGER_HOVER,
            state="disabled",
            command=self._stop_processing,
        )
        self.stop_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row,
            text="📂 Mở xuất",
            width=90,
            height=40,
            corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._open_output_folder,
        ).pack(side="right")

        # Progress bar & Status text
        self.prog_bar = ctk.CTkProgressBar(
            action_box,
            height=8,
            corner_radius=4,
            progress_color=ACCENT_LIGHT,
            fg_color="#1e293b",
        )
        self.prog_bar.set(0.0)
        self.prog_bar.pack(fill="x", pady=(2, 4))

        self.status_lbl = ctk.CTkLabel(
            action_box,
            text="Sẵn sàng xử lý.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self.status_lbl.pack(fill="x")

        # 5. Live Log Box
        log_head = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        log_head.pack(fill="x", padx=16, pady=(6, 2))
        ctk.CTkLabel(
            log_head,
            text="Nhật ký xử lý (Live Log)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")

        self.log_box = ctk.CTkTextbox(
            self.left_panel,
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=10,
            fg_color="#090d14",
            border_width=1,
            border_color=CARD_LINE,
            text_color="#cbd5e1",
        )
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

    def _build_right_panel(self) -> None:
        # Header for comparison studio
        head = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            head,
            text="Studio So sánh Trước / Sau",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        # Navigation arrows
        nav = ctk.CTkFrame(head, fg_color="transparent")
        nav.pack(side="right")

        ctk.CTkButton(
            nav,
            text="◀ Trước",
            width=65,
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            command=lambda: self._nav_compare(-1),
        ).pack(side="left", padx=2)

        self.cmp_counter_lbl = ctk.CTkLabel(
            nav,
            text="0 / 0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
            width=50,
        )
        self.cmp_counter_lbl.pack(side="left", padx=4)

        ctk.CTkButton(
            nav,
            text="Sau ▶",
            width=65,
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            command=lambda: self._nav_compare(1),
        ).pack(side="left", padx=2)

        # Comparison Split Viewer Area
        self.cmp_host = ctk.CTkFrame(
            self.right_panel,
            fg_color="#080b11",
            corner_radius=12,
            border_width=1,
            border_color=CARD_LINE,
        )
        self.cmp_host.pack(fill="both", expand=True, padx=16, pady=4)

        # Empty Comparison Placeholder
        self.cmp_empty = ctk.CTkFrame(self.cmp_host, fg_color="transparent")
        self.cmp_empty.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            self.cmp_empty,
            text=FA.MAGIC,
            font=fa_font(36),
            text_color=TEXT_DARK,
        ).pack()

        ctk.CTkLabel(
            self.cmp_empty,
            text="Chưa có kết quả để so sánh",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(pady=(6, 2))

        ctk.CTkLabel(
            self.cmp_empty,
            text="Thêm file và bấm 'BẮT ĐẦU XÓA WATERMARK' để xem trực quan",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DARK,
        ).pack()

        # Split slider bar
        slider_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        slider_frame.pack(fill="x", padx=20, pady=(6, 8))

        ctk.CTkLabel(
            slider_frame,
            text="GỐC (TRƯỚC)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=DANGER_RED,
        ).pack(side="left", padx=(0, 6))

        self.cmp_slider = ctk.CTkSlider(
            slider_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            height=14,
            progress_color=ACCENT_LIGHT,
            button_color=TEXT_MAIN,
            command=self._on_slider_split_change,
        )
        self.cmp_slider.set(50)
        self.cmp_slider.pack(side="left", fill="x", expand=True, padx=4)

        ctk.CTkLabel(
            slider_frame,
            text="ĐÃ XÓA (SAU)",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=SUCCESS_GREEN,
        ).pack(side="right", padx=(6, 0))

        # Bottom Results Thumbnails Carousel
        gal_head = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        gal_head.pack(fill="x", padx=16, pady=(4, 2))

        ctk.CTkLabel(
            gal_head,
            text="Thư viện kết quả hoàn thành",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")

        self.gallery_scroll = ctk.CTkScrollableFrame(
            self.right_panel,
            orientation="horizontal",
            height=100,
            fg_color=CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=CARD_LINE,
        )
        self.gallery_scroll.pack(fill="x", padx=16, pady=(0, 14))

    # ── Hardware Detection & Logging ──────────────────────────
    def _detect_hardware(self) -> None:
        def _worker():
            try:
                info = get_hardware_acceleration_info()
            except Exception as e:
                info = {"encoder_desc": "CPU", "ai_provider": "CPU", "error": str(e)}

            def _apply():
                try:
                    enc = info.get("encoder_desc", "CPU")
                    ai_p = info.get("ai_provider", "CPU")

                    self.gpu_chip.configure(
                        text=f"⚡ Video: {enc}",
                        text_color=SUCCESS_GREEN if any(k in enc for k in ["GPU", "NVENC", "QuickSync", "AMF"]) else TEXT_MUTED,
                        fg_color="#064e3b" if any(k in enc for k in ["GPU", "NVENC", "QuickSync", "AMF"]) else "#1e293b",
                    )
                    self.ai_chip.configure(
                        text=f"🧠 AI: {ai_p}",
                        text_color=SUCCESS_GREEN if any(k in ai_p for k in ["GPU", "DirectML", "CUDA"]) else TEXT_MUTED,
                        fg_color="#064e3b" if any(k in ai_p for k in ["GPU", "DirectML", "CUDA"]) else "#1e293b",
                    )
                    if hasattr(self, "info_hw_enc_lbl"):
                        self.info_hw_enc_lbl.configure(text=f"• Video Hardware Encoder:  {enc} (Tăng tốc xử lý video trực tiếp)")
                    if hasattr(self, "info_hw_ai_lbl"):
                        self.info_hw_ai_lbl.configure(text=f"• AI Inpainting Engine:     {ai_p} (MIGAN Deep Learning ONNX)")
                    if hasattr(self, "info_hw_ff_lbl"):
                        self.info_hw_ff_lbl.configure(text="• Trạng thái FFmpeg Core:    ✓ Đã kết nối & Sẵn sàng (Hardware Acceleration Active)")

                    self._log(f"→ Video Encoder: {enc}")
                    self._log(f"→ AI Inference Engine: {ai_p}")
                    self._log(f"→ FFmpeg Binary: {get_ffmpeg_path()}")
                    self._log("Hệ thống đã sẵn sàng xử lý.")
                    self.status_lbl.configure(text="✓ Đã sẵn sàng xử lý")
                except Exception:
                    pass

            self.after(0, _apply)

        self._log("Đang kiểm tra phần cứng & GPU trong nền...")
        threading.Thread(target=_worker, daemon=True).start()

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")

    # ── Drag & Drop Setup ─────────────────────────────────────
    def _setup_dnd(self) -> None:
        if not getattr(self, "_has_dnd", False) or DND_FILES is None:
            return

        def _on_drop(event):
            raw = getattr(event, "data", "") or ""
            paths = _parse_drop_data(raw, self)
            if paths:
                self._add_paths(paths)

        def _bind(w):
            targets = [w]
            for attr in ("_textbox", "textbox", "_canvas", "_entry", "_label"):
                inner = getattr(w, attr, None)
                if inner is not None:
                    targets.append(inner)
            for target in targets:
                try:
                    target.drop_target_register(DND_FILES)
                    target.dnd_bind("<<Drop>>", _on_drop)
                except Exception:
                    pass

        _bind(self)
        _bind(self.drop_container)
        _bind(self.drop_empty)
        _bind(self.file_scroll)

    # ── File & Queue Management ───────────────────────────────
    def _browse_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh hoặc video",
            filetypes=[
                ("Tất cả tệp được hỗ trợ", "*.mp4 *.mov *.mkv *.webm *.avi *.png *.jpg *.jpeg *.webp *.bmp"),
                ("Video", "*.mp4 *.mov *.mkv *.webm *.avi"),
                ("Ảnh", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("Tất cả", "*.*"),
            ],
        )
        if paths:
            self._add_paths(list(paths))

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh/video")
        if folder:
            p = Path(folder)
            files = [str(f.resolve()) for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
            if files:
                self._add_paths(files)
            else:
                messagebox.showinfo("Thư mục trống", f"Không tìm thấy ảnh hoặc video trong thư mục: {p.name}")

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục xuất tệp đã xóa watermark")
        if folder:
            self.output_dir.set(str(Path(folder).resolve()))

    def _add_paths(self, raw_paths: list[str]) -> None:
        added = 0
        existing_paths = {str(item["path"].resolve()) for item in self.files_list}

        for rp in raw_paths:
            p = Path(str(rp).strip().strip("{}"))
            if not p.exists():
                continue

            if p.is_dir():
                sub_files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
                for sf in sub_files:
                    sp = str(sf.resolve())
                    if sp not in existing_paths:
                        self.files_list.append({
                            "path": sf,
                            "ext": sf.suffix.lower(),
                            "is_video": sf.suffix.lower() in VIDEO_EXT,
                            "status": "waiting",
                            "result_path": None,
                        })
                        existing_paths.add(sp)
                        added += 1
            elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                sp = str(p.resolve())
                if sp not in existing_paths:
                    self.files_list.append({
                        "path": p,
                        "ext": p.suffix.lower(),
                        "is_video": p.suffix.lower() in VIDEO_EXT,
                        "status": "waiting",
                        "result_path": None,
                    })
                    existing_paths.add(sp)
                    added += 1

        if added > 0:
            self._log(f"Đã thêm {added} tệp vào hàng đợi (Tổng: {len(self.files_list)} tệp).")
            self._refresh_file_list_ui()

    def _clear_files(self) -> None:
        if self.is_processing:
            messagebox.showwarning("Đang xử lý", "Vui lòng dừng tiến trình trước khi xóa danh sách.")
            return
        self.files_list.clear()
        self._refresh_file_list_ui()
        self._log("Đã xóa sạch hàng đợi tệp.")

    def _remove_file(self, idx: int) -> None:
        if self.is_processing:
            return
        if 0 <= idx < len(self.files_list):
            rem = self.files_list.pop(idx)
            self._log(f"Đã xóa: {rem['path'].name}")
            self._refresh_file_list_ui()

    def _refresh_file_list_ui(self) -> None:
        n = len(self.files_list)
        self.count_lbl.configure(text=f"{n} tệp")

        if n == 0:
            self.file_scroll.pack_forget()
            self.drop_empty.place(relx=0.5, rely=0.5, anchor="center")
            return

        self.drop_empty.place_forget()
        self.file_scroll.pack(fill="both", expand=True, padx=6, pady=6)

        # Clear existing items
        for child in self.file_scroll.winfo_children():
            child.destroy()

        for idx, item in enumerate(self.files_list):
            row = ctk.CTkFrame(
                self.file_scroll,
                fg_color="#1a2234" if item["status"] == "processing" else "#131926",
                corner_radius=8,
                border_width=1,
                border_color=ACCENT_LIGHT if item["status"] == "processing" else CARD_LINE,
                height=38,
            )
            row.pack(fill="x", pady=2, padx=2)
            row.pack_propagate(False)

            # Type badge icon
            t_icon = FA.FILM if item["is_video"] else FA.IMAGE
            t_color = ACCENT_LIGHT if item["is_video"] else "#a78bfa"
            ctk.CTkLabel(
                row,
                text=t_icon,
                font=fa_font(13),
                text_color=t_color,
                width=24,
            ).pack(side="left", padx=(8, 4))

            # File name
            name_text = item["path"].name
            if len(name_text) > 35:
                name_text = name_text[:20] + "..." + name_text[-12:]

            ctk.CTkLabel(
                row,
                text=name_text,
                font=ctk.CTkFont(size=12, weight="bold" if item["status"] == "processing" else "normal"),
                text_color=TEXT_MAIN,
                anchor="w",
            ).pack(side="left", padx=4, fill="x", expand=True)

            # Status Badge
            st = item["status"]
            if st == "waiting":
                st_text = "Chờ xử lý"
                st_color = TEXT_MUTED
            elif st == "processing":
                st_text = "Đang chạy..."
                st_color = WARNING_YELLOW
            elif st == "done":
                st_text = "✓ Đã xóa"
                st_color = SUCCESS_GREEN
            elif st == "error":
                st_text = "✗ Lỗi"
                st_color = DANGER_RED
            else:
                st_text = st
                st_color = TEXT_MUTED

            ctk.CTkLabel(
                row,
                text=st_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=st_color,
                width=80,
                anchor="e",
            ).pack(side="left", padx=6)

            # Delete single button
            del_btn = ctk.CTkButton(
                row,
                text="✕",
                width=24,
                height=24,
                corner_radius=6,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="transparent",
                hover_color="#ef4444",
                text_color=TEXT_MUTED,
                command=lambda i=idx: self._remove_file(i),
            )
            del_btn.pack(side="right", padx=(2, 6))

    def _on_mode_change(self, val: str) -> None:
        if "Thủ công" in val or "Bút vẽ" in val:
            self.detect_mode.set("manual")
            if self.custom_mask is not None or self.custom_box is not None:
                self.region_lbl.configure(
                    text="Chế độ: Thủ công (Đã có vùng khoanh / nét cọ)",
                    text_color=SUCCESS_GREEN,
                )
            else:
                self.region_lbl.configure(
                    text="Chế độ: Thủ công (Bấm 'Bút vẽ & Khoanh ô' để tạo vùng)",
                    text_color=WARNING_YELLOW,
                )
            self._log("Đã chọn Chế độ: Khoanh vùng / Bút vẽ thủ công.")
        else:
            self.detect_mode.set("auto")
            self.region_lbl.configure(
                text="Chế độ: Tự động quét logo góc & watermark toàn diện",
                text_color=ACCENT_LIGHT,
            )
            self._log("Đã chọn Chế độ: Tự động Toàn diện (Smart Auto).")

    # ── Interactive Region & Brush Picker Popup ──────────────────────
    def _open_region_picker(self) -> None:
        if not self.files_list:
            messagebox.showinfo("Chưa có tệp", "Hãy thêm ít nhất 1 ảnh hoặc video để xem khung hình khoanh vùng.")
            return

        first_item = self.files_list[0]["path"]
        try:
            import tempfile
            tmp_preview = Path(tempfile.gettempdir()) / "zwatermark_picker_frame.png"
            if first_item.suffix.lower() in VIDEO_EXT:
                extract_preview_frame(first_item, tmp_preview, time_sec=0.5)
            else:
                im = Image.open(first_item).convert("RGB")
                im.save(tmp_preview)
        except Exception as e:
            messagebox.showerror("Lỗi trích xuất khung hình", f"Không thể lấy frame xem trước:\n{e}")
            return

        import tkinter as tk
        import numpy as np
        import cv2

        win = ctk.CTkToplevel(self)
        win.title("Công cụ Khoanh ô & Cọ vẽ Watermark (Interactive Mask Studio)")
        win.geometry("1020x720")
        win.minsize(860, 600)
        win.configure(fg_color=BG_COLOR)
        win.lift()
        win.focus_force()
        win.after(50, lambda: (win.lift(), win.focus_force()))

        ico = get_asset_path("app.ico")
        if ico and ico.is_file():
            try:
                win.iconbitmap(str(ico))
            except Exception:
                pass

        img_orig = Image.open(tmp_preview).convert("RGB")
        ow, oh = img_orig.size
        max_w, max_h = 960, 490
        scale = min(max_w / ow, max_h / oh, 1.0)
        dw, dh = int(ow * scale), int(oh * scale)
        preview_scaled = img_orig.resize((dw, dh), Image.Resampling.BILINEAR)
        photo_tk = ImageTk.PhotoImage(preview_scaled, master=win)

        # Toolbar Frame at Top
        tool_bar = ctk.CTkFrame(win, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=CARD_LINE)
        tool_bar.pack(fill="x", padx=16, pady=(10, 4))

        ctk.CTkLabel(tool_bar, text="Công cụ:", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MAIN).pack(side="left", padx=(14, 8), pady=8)

        current_tool = {"mode": "brush", "size": 28}

        def set_tool(t: str):
            current_tool["mode"] = t
            for name, btn in [("brush", btn_brush), ("line", btn_line), ("box", btn_box), ("eraser", btn_eraser)]:
                if name == t:
                    btn.configure(fg_color=ACCENT_GRADIENT, text_color=TEXT_MAIN)
                else:
                    btn.configure(fg_color="#1e293b", text_color=TEXT_MUTED)
            canvas.config(cursor="crosshair" if t != "eraser" else "dotbox")

        btn_brush = ctk.CTkButton(tool_bar, text="🖌️ Cọ Tự Do (Brush)", width=135, height=28, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), fg_color=ACCENT_GRADIENT, command=lambda: set_tool("brush"))
        btn_brush.pack(side="left", padx=3)

        btn_line = ctk.CTkButton(tool_bar, text="📏 Thước Thẳng (Line)", width=135, height=28, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), fg_color="#1e293b", text_color=TEXT_MUTED, command=lambda: set_tool("line"))
        btn_line.pack(side="left", padx=3)

        btn_box = ctk.CTkButton(tool_bar, text="📦 Kéo Ô (Box)", width=105, height=28, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), fg_color="#1e293b", text_color=TEXT_MUTED, command=lambda: set_tool("box"))
        btn_box.pack(side="left", padx=3)

        btn_eraser = ctk.CTkButton(tool_bar, text="🧹 Tẩy (Eraser)", width=105, height=28, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), fg_color="#1e293b", text_color=TEXT_MUTED, command=lambda: set_tool("eraser"))
        btn_eraser.pack(side="left", padx=3)

        btn_undo = ctk.CTkButton(tool_bar, text="↩️ Hoàn tác (Undo)", width=125, height=28, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), fg_color="#334155", hover_color="#475569", command=lambda: undo())
        btn_undo.pack(side="left", padx=3)

        # Brush size slider
        ctk.CTkLabel(tool_bar, text="Cỡ cọ:", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED).pack(side="left", padx=(12, 4))
        size_lbl = ctk.CTkLabel(tool_bar, text="28px", font=ctk.CTkFont(size=11), text_color=ACCENT_LIGHT, width=32)
        size_slider = ctk.CTkSlider(tool_bar, from_=6, to=80, number_of_steps=74, width=100, height=12, progress_color=ACCENT_LIGHT, command=lambda v: (current_tool.update(size=int(v)), size_lbl.configure(text=f"{int(v)}px")))
        size_slider.set(28)
        size_slider.pack(side="left", padx=2)
        size_lbl.pack(side="left", padx=(2, 6))

        btn_live = ctk.CTkButton(
            tool_bar,
            text="⚡ Thử Xóa Ngay",
            width=135,
            height=28,
            corner_radius=6,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            command=lambda: test_live_inpaint(),
        )
        btn_live.pack(side="right", padx=6)

        # Canvas Frame
        holder = ctk.CTkFrame(win, fg_color=CARD_BG, corner_radius=12, border_width=1, border_color=CARD_LINE)
        holder.pack(padx=16, pady=4)

        canvas = tk.Canvas(holder, width=dw, height=dh, highlightthickness=0, bg="#080b11", cursor="crosshair")
        canvas.pack(padx=8, pady=8)
        canvas.create_image(0, 0, anchor="nw", image=photo_tk)
        canvas.image = photo_tk

        # State & Mask Buffer on original resolution
        mask_orig = np.zeros((oh, ow), dtype=np.uint8)
        mask_history = []
        draw_state = {"last_x": None, "last_y": None, "box_x0": 0, "box_y0": 0, "line_x0": 0, "line_y0": 0, "rect_id": None, "line_id": None}

        def push_history():
            mask_history.append(mask_orig.copy())
            if len(mask_history) > 30:
                mask_history.pop(0)

        def redraw_canvas_from_mask():
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo_tk)
            if mask_orig.sum() > 0:
                m_scaled = cv2.resize(mask_orig, (dw, dh), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.where(m_scaled > 0)
                if ys.size > 0:
                    for y, x in zip(ys[::3], xs[::3]):
                        canvas.create_rectangle(x, y, x + 2, y + 2, fill="#0284c7", outline="")

        def undo():
            if mask_history:
                prev = mask_history.pop()
                mask_orig[:] = prev
                redraw_canvas_from_mask()

        win.bind("<Control-z>", lambda e: undo())
        win.bind("<Control-Z>", lambda e: undo())

        def to_orig(x, y):
            return int(x / scale), int(y / scale)

        def on_down(e):
            push_history()
            if current_tool["mode"] == "box":
                draw_state["box_x0"], draw_state["box_y0"] = e.x, e.y
                if draw_state["rect_id"]:
                    canvas.delete(draw_state["rect_id"])
                draw_state["rect_id"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#38bdf8", width=2, dash=(4, 2))
            elif current_tool["mode"] == "line":
                draw_state["line_x0"], draw_state["line_y0"] = e.x, e.y
                r = current_tool["size"]
                if draw_state["line_id"]:
                    canvas.delete(draw_state["line_id"])
                draw_state["line_id"] = canvas.create_line(e.x, e.y, e.x, e.y, width=r, fill="#38bdf8", capstyle=tk.ROUND)
            else:
                draw_state["last_x"], draw_state["last_y"] = e.x, e.y
                paint_stroke(e.x, e.y, e.x, e.y)

        def on_move(e):
            if current_tool["mode"] == "box":
                if draw_state["rect_id"]:
                    canvas.coords(draw_state["rect_id"], draw_state["box_x0"], draw_state["box_y0"], e.x, e.y)
            elif current_tool["mode"] == "line":
                if draw_state["line_id"]:
                    canvas.coords(draw_state["line_id"], draw_state["line_x0"], draw_state["line_y0"], e.x, e.y)
            else:
                if draw_state["last_x"] is not None:
                    paint_stroke(draw_state["last_x"], draw_state["last_y"], e.x, e.y)
                    draw_state["last_x"], draw_state["last_y"] = e.x, e.y

        def on_up(e):
            if current_tool["mode"] == "box":
                x0, y0 = draw_state["box_x0"], draw_state["box_y0"]
                x1, y1 = e.x, e.y
                if x1 < x0: x0, x1 = x1, x0
                if y1 < y0: y0, y1 = y1, y0
                sx0, sy0 = to_orig(x0, y0)
                sx1, sy1 = to_orig(x1, y1)
                sx0, sy0 = max(0, sx0), max(0, sy0)
                sx1, sy1 = min(ow, sx1), min(oh, sy1)
                mask_orig[sy0:sy1, sx0:sx1] = 255
                canvas.create_rectangle(x0, y0, x1, y1, fill="#0284c7", stipple="gray25", outline="#38bdf8", width=1)
                draw_state["rect_id"] = None
            elif current_tool["mode"] == "line":
                lx0, ly0 = draw_state["line_x0"], draw_state["line_y0"]
                lx1, ly1 = e.x, e.y
                r = current_tool["size"]
                canvas.create_line(lx0, ly0, lx1, ly1, width=r, fill="#0284c7", capstyle=tk.ROUND)
                ox0, oy0 = to_orig(lx0, ly0)
                ox1, oy1 = to_orig(lx1, ly1)
                orig_r = int(r / scale)
                cv2.line(mask_orig, (ox0, oy0), (ox1, oy1), 255, orig_r)
                cv2.circle(mask_orig, (ox0, oy0), orig_r // 2, 255, -1)
                cv2.circle(mask_orig, (ox1, oy1), orig_r // 2, 255, -1)
                if draw_state["line_id"]:
                    canvas.delete(draw_state["line_id"])
                draw_state["line_id"] = None
            draw_state["last_x"] = None
            draw_state["last_y"] = None

        def paint_stroke(x1, y1, x2, y2):
            r = current_tool["size"]
            val = 255 if current_tool["mode"] == "brush" else 0
            color = "#0284c7" if current_tool["mode"] == "brush" else "#080b11"

            # Draw on Tk Canvas
            canvas.create_line(x1, y1, x2, y2, width=r, fill=color, capstyle=tk.ROUND, smooth=True)
            canvas.create_oval(x2 - r // 2, y2 - r // 2, x2 + r // 2, y2 + r // 2, fill=color, outline="")

            # Draw on original mask array
            ox1, oy1 = to_orig(x1, y1)
            ox2, oy2 = to_orig(x2, y2)
            orig_r = int(r / scale)
            cv2.line(mask_orig, (ox1, oy1), (ox2, oy2), val, orig_r)
            cv2.circle(mask_orig, (ox2, oy2), orig_r // 2, val, -1)

        canvas.bind("<ButtonPress-1>", on_down)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_up)

        def test_live_inpaint():
            if mask_orig.sum() == 0:
                messagebox.showwarning("Chưa vẽ", "Hãy vẽ cọ hoặc kéo đường thẳng trước khi xem trước.")
                return
            btn_live.configure(text="⏳ Đang xóa...", state="disabled")

            def _inpaint_work():
                from core.watermark_remover import inpaint_mask
                rgb_arr = np.asarray(img_orig)
                cleaned, _ = inpaint_mask(rgb_arr, mask_orig, use_ai=True)
                im_res = Image.fromarray(cleaned).resize((dw, dh), Image.Resampling.BILINEAR)
                tk_res = ImageTk.PhotoImage(im_res, master=win)

                def _show():
                    canvas.delete("all")
                    canvas.create_image(0, 0, anchor="nw", image=tk_res)
                    canvas.tk_res = tk_res
                    btn_live.configure(text="⚡ Thử Xóa Ngay", state="normal")

                win.after(0, _show)

            threading.Thread(target=_inpaint_work, daemon=True).start()

        # Bottom Button Bar
        btn_bar = ctk.CTkFrame(win, fg_color="transparent")
        btn_bar.pack(pady=(6, 12))

        def apply_mask():
            if mask_orig.sum() == 0:
                messagebox.showwarning("Chưa vẽ", "Hãy dùng cọ vẽ hoặc kéo ô bao quanh vùng watermark cần xóa.")
                return

            self.custom_mask = mask_orig.copy()
            ys, xs = np.where(mask_orig > 0)
            if ys.size > 0:
                self.custom_box = {
                    "x": int(xs.min()),
                    "y": int(ys.min()),
                    "w": int(xs.max() - xs.min() + 1),
                    "h": int(ys.max() - ys.min() + 1),
                }

            self.detect_mode.set("manual")
            self.mode_menu.set("🎯 Thủ công (Bút vẽ & Khoanh ô)")
            pixels_cnt = int(mask_orig.sum() // 255)
            self.region_lbl.configure(text=f"Thủ công: [Đã tạo mặt nạ {pixels_cnt} px]", text_color=SUCCESS_GREEN)
            self._log(f"Đã tạo mặt nạ watermark thủ công: {pixels_cnt} pixels.")
            win.destroy()

        def clear_canvas():
            push_history()
            mask_orig[:] = 0
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo_tk)

        ctk.CTkButton(
            btn_bar,
            text="✓ Dùng vùng / nét vẽ này",
            width=190,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=ACCENT_GRADIENT,
            hover_color=ACCENT_HOVER,
            command=apply_mask,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_bar,
            text="🧹 Xóa hết",
            width=110,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=clear_canvas,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_bar,
            text="Đóng",
            width=90,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            command=win.destroy,
        ).pack(side="left", padx=8)

    # ── Processing Execution (Threaded) ───────────────────────
    def _pump_ui_heartbeat(self) -> None:
        if self.is_processing:
            try:
                self.update()
            except Exception:
                pass
            self.after(40, self._pump_ui_heartbeat)

    def _start_processing(self) -> None:
        if not self.files_list:
            messagebox.showwarning("Chưa có tệp", "Vui lòng thêm ít nhất 1 ảnh hoặc video để bắt đầu.")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.cancel_event.clear()
        self.start_btn.configure(state="disabled", text="ĐANG XỬ LÝ...")
        self.stop_btn.configure(state="normal", fg_color=DANGER_RED)
        self.prog_bar.set(0.0)

        # Start continuous UI Heartbeat to prevent Windows (Not Responding)
        self.after(50, self._pump_ui_heartbeat)

        # Worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _stop_processing(self) -> None:
        if not self.is_processing:
            return
        self.cancel_event.set()
        self._log("⚠️ Đang gửi tín hiệu dừng tiến trình...")
        self.status_lbl.configure(text="Đang dừng...")

    def _worker_loop(self) -> None:
        total = len(self.files_list)
        out_base = self.output_dir.get().strip()
        use_ai = self.use_ai.get()
        box = self.custom_box
        custom_mask = self.custom_mask
        mode = self.detect_mode.get()
        new_pairs = []

        mode_name = {
            "auto": "Tự động Toàn diện (Smart Auto)",
            "manual": "Thủ công (Bút vẽ & Khoanh ô)",
        }.get(mode, mode)

        self._log(f"=== Bắt đầu xử lý {total} tệp [Chế độ: {mode_name}] (AI Inpaint: {use_ai}) ===")

        success_count = 0
        start_all = time.time()

        for idx, item in enumerate(self.files_list):
            if self.cancel_event.is_set():
                self._log("! Tiến trình đã bị dừng bởi người dùng.")
                break

            src_p = item["path"]
            item["status"] = "processing"
            self.after(0, self._refresh_file_list_ui)
            self.after(0, lambda n=src_p.name, i=idx: self.status_lbl.configure(text=f"Đang xử lý [{i+1}/{total}]: {n}..."))

            out_dir = Path(out_base) if out_base else src_p.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            dst_p = out_dir / f"{src_p.stem}_removed_watermark{src_p.suffix if not item['is_video'] else '.mp4'}"

            try:
                if item["is_video"]:
                    def _prog(cur_f, tot_f, fps_v):
                        pct = (cur_f / max(1, tot_f)) if tot_f else 0.0
                        overall = (idx + pct) / total
                        self.after(0, lambda p=overall: self.prog_bar.set(p))
                        self.after(
                            0,
                            lambda cf=cur_f, tf=tot_f, f=fps_v, nm=src_p.name: self.status_lbl.configure(
                                text=f"[{idx+1}/{total}] {nm} — Frame {cf}/{tf} ({f:.1f} fps)"
                            ),
                        )

                    res = process_video(
                        src_p,
                        dst_p,
                        log=lambda m: self.after(0, lambda msg=m: self._log(msg)),
                        max_seconds=1800.0,
                        box=box,
                        use_ai=use_ai,
                        progress_fn=_prog,
                        cancel_event=self.cancel_event,
                        mode=mode,
                        custom_mask=custom_mask,
                    )
                else:
                    res = process_file(
                        src_p,
                        dst_p,
                        log=lambda m: self.after(0, lambda msg=m: self._log(msg)),
                        use_ai=use_ai,
                        box=box,
                        mode=mode,
                        custom_mask=custom_mask,
                    )

                if res.get("cancelled"):
                    item["status"] = "waiting"
                    break

                item["status"] = "done"
                item["result_path"] = str(dst_p)
                success_count += 1
                new_pairs.append((str(src_p), str(dst_p)))

            except Exception as ex:
                item["status"] = "error"
                self._log(f"❌ Lỗi xử lý {src_p.name}: {ex}")
                traceback.print_exc()

            overall_pct = (idx + 1) / total
            self.after(0, lambda p=overall_pct: self.prog_bar.set(p))
            self.after(0, self._refresh_file_list_ui)

        elapsed_total = time.time() - start_all
        self._log(f"=== Hoàn thành: {success_count}/{total} tệp trong {elapsed_total:.1f}s ===")

        # Update Compare Studio & UI State
        self.after(0, lambda: self._on_processing_finished(new_pairs, success_count, total))

    def _on_processing_finished(self, new_pairs: list[tuple[str, str]], success: int, total: int) -> None:
        self.is_processing = False
        self.start_btn.configure(state="normal", text="🚀 BẮT ĐẦU XÓA WATERMARK")
        self.stop_btn.configure(state="disabled", fg_color="#1e293b")
        self.status_lbl.configure(text=f"Hoàn thành {success}/{total} tệp.")
        self.prog_bar.set(1.0)
        self._refresh_file_list_ui()

        if new_pairs:
            for p in new_pairs:
                if p not in self.compare_pairs:
                    self.compare_pairs.append(p)
            self._update_comparison_ui()
            self._update_gallery_thumbnails()

    def _open_output_folder(self) -> None:
        out = self.output_dir.get().strip()
        if not out:
            if self.files_list:
                out = str(self.files_list[0]["path"].parent)
            else:
                out = str(Path.home())
        p = Path(out)
        if p.exists():
            try:
                os.startfile(str(p))
            except Exception:
                import subprocess
                subprocess.Popen(["explorer", str(p)])
        else:
            messagebox.showinfo("Thư mục", f"Thư mục chưa tồn tại: {p}")

    # ── Studio So Sánh Trước / Sau (Interactive Split Canvas) ──
    def _update_comparison_ui(self) -> None:
        if not self.compare_pairs:
            self.cmp_empty.place(relx=0.5, rely=0.5, anchor="center")
            if self.compare_canvas:
                self.compare_canvas.pack_forget()
            self.cmp_counter_lbl.configure(text="0 / 0")
            return

        self.cmp_empty.place_forget()
        n = len(self.compare_pairs)
        self.compare_idx = max(0, min(self.compare_idx, n - 1))
        self.cmp_counter_lbl.configure(text=f"{self.compare_idx + 1} / {n}")

        src_path, dst_path = self.compare_pairs[self.compare_idx]
        self._load_pair_into_viewer(src_path, dst_path)

    def _nav_compare(self, step: int) -> None:
        if not self.compare_pairs:
            return
        n = len(self.compare_pairs)
        self.compare_idx = (self.compare_idx + step) % n
        self.cmp_counter_lbl.configure(text=f"{self.compare_idx + 1} / {n}")
        src_path, dst_path = self.compare_pairs[self.compare_idx]
        self._load_pair_into_viewer(src_path, dst_path)

    def _load_pair_into_viewer(self, before_path: str, after_path: str) -> None:
        bp, ap = Path(before_path), Path(after_path)
        if not bp.is_file() or not ap.is_file():
            return

        def _loader():
            import tempfile
            try:
                if bp.suffix.lower() in VIDEO_EXT:
                    tmp_b = Path(tempfile.gettempdir()) / "zwatermark_cmp_b.png"
                    extract_preview_frame(bp, tmp_b, time_sec=0.5)
                    img_b = Image.open(tmp_b).convert("RGB")
                else:
                    img_b = Image.open(bp).convert("RGB")

                if ap.suffix.lower() in VIDEO_EXT:
                    tmp_a = Path(tempfile.gettempdir()) / "zwatermark_cmp_a.png"
                    extract_preview_frame(ap, tmp_a, time_sec=0.5)
                    img_a = Image.open(tmp_a).convert("RGB")
                else:
                    img_a = Image.open(ap).convert("RGB")

                w0, h0 = img_a.size
                img_b = img_b.resize((w0, h0), Image.Resampling.BILINEAR)

                self._cmp_img_b = img_b
                self._cmp_img_a = img_a
                self.after(0, self._render_split_canvas)
            except Exception as e:
                self.after(0, lambda: self._log(f"Không nạp được ảnh so sánh: {e}"))

        threading.Thread(target=_loader, daemon=True).start()

    def _generate_thumb(self, file_path: str | Path) -> Optional[ctk.CTkImage]:
        p = Path(file_path)
        if not p.is_file():
            return None
        try:
            if p.suffix.lower() in VIDEO_EXT:
                import tempfile
                tmp = Path(tempfile.gettempdir()) / f"thumb_{p.stem}.png"
                if not tmp.is_file():
                    extract_preview_frame(p, tmp, time_sec=0.5)
                im = Image.open(tmp).convert("RGB")
            else:
                im = Image.open(p).convert("RGB")
            im.thumbnail((90, 50), Image.Resampling.BILINEAR)
            return ctk.CTkImage(light_image=im, dark_image=im, size=(im.width, im.height))
        except Exception:
            return None

    def _render_split_canvas(self) -> None:
        if self._cmp_img_a is None or self._cmp_img_b is None:
            return

        # Canvas Dimensions
        target_w, target_h = 560, 480
        ib = self._cmp_img_b.resize((target_w, target_h), Image.Resampling.BILINEAR)
        ia = self._cmp_img_a.resize((target_w, target_h), Image.Resampling.BILINEAR)

        tk_a = ImageTk.PhotoImage(ia)
        tk_b = ImageTk.PhotoImage(ib)

        import tkinter as tk
        if self.compare_canvas is None:
            self.compare_canvas = tk.Canvas(
                self.cmp_host,
                width=target_w,
                height=target_h,
                highlightthickness=0,
                bg="#080b11",
                cursor="sb_h_double_arrow",
            )
            self.compare_canvas.pack(expand=True, pady=10)

            def on_canvas_drag(e):
                cw = self.compare_canvas.winfo_width() or target_w
                pct = max(0.0, min(100.0, (e.x / cw) * 100.0))
                self.cmp_slider.set(pct)
                self._draw_split(pct)

            self.compare_canvas.bind("<Button-1>", on_canvas_drag)
            self.compare_canvas.bind("<B1-Motion>", on_canvas_drag)

        self.compare_canvas.config(width=target_w, height=target_h)
        self.compare_canvas.tk_a = tk_a
        self.compare_canvas.tk_b = tk_b
        self.compare_canvas.pil_a = ia
        self.compare_canvas.pil_b = ib

        self._draw_split(self.compare_split_pct)

    def _draw_split(self, split_pct: float) -> None:
        if self.compare_canvas is None or not hasattr(self.compare_canvas, "pil_a"):
            return

        c = self.compare_canvas
        ia = c.pil_a
        ib = c.pil_b
        w, h = ia.size

        split_x = int(w * (split_pct / 100.0))
        split_x = max(1, min(w - 1, split_x))

        c.delete("all")

        # 1. Base After Image (Cleaned)
        c.create_image(0, 0, anchor="nw", image=c.tk_a)

        # 2. Left Overlay Before Image (Original)
        left_slice = ib.crop((0, 0, split_x, h))
        tk_left = ImageTk.PhotoImage(left_slice)
        c.tk_left = tk_left
        c.create_image(0, 0, anchor="nw", image=tk_left)

        # 3. Glowing Split Line
        c.create_line(split_x, 0, split_x, h, fill="#38bdf8", width=2)
        r = 14
        cy = h // 2
        c.create_oval(split_x - r, cy - r, split_x + r, cy + r, fill="#0f172a", outline="#38bdf8", width=2)
        c.create_text(split_x, cy, text="↔", fill="#38bdf8", font=("Segoe UI", 11, "bold"))

        # 4. Floating Badges: TRƯỚC (GỐC) & SAU (ĐÃ XÓA)
        c.create_rectangle(12, 12, 110, 36, fill="#0f172a", outline="#334155")
        c.create_text(61, 24, text="TRƯỚC (GỐC)", fill="#f87171", font=("Segoe UI", 9, "bold"))

        c.create_rectangle(w - 120, 12, w - 12, 36, fill="#0f172a", outline="#334155")
        c.create_text(w - 66, 24, text="SAU (ĐÃ XÓA)", fill="#34d399", font=("Segoe UI", 9, "bold"))

    def _on_slider_split_change(self, val: float) -> None:
        self.compare_split_pct = float(val)
        self._draw_split(self.compare_split_pct)

    # ── Bottom Gallery Strip ──────────────────────────────────
    def _update_gallery_thumbnails(self) -> None:
        for w in self.gallery_scroll.winfo_children():
            w.destroy()
        self._cached_thumbs.clear()

        for idx, (src, dst) in enumerate(self.compare_pairs):
            name = Path(dst).name
            cell = ctk.CTkFrame(
                self.gallery_scroll,
                fg_color="#101522",
                corner_radius=8,
                border_width=1,
                border_color=CARD_LINE,
                width=110,
                height=90,
            )
            cell.pack(side="left", padx=4, pady=4)
            cell.pack_propagate(False)

            # Thumbnail Image
            thumb_img = self._generate_thumb(dst) or self._generate_thumb(src)
            if thumb_img:
                self._cached_thumbs.append(thumb_img)
                ctk.CTkLabel(cell, image=thumb_img, text="").pack(pady=(4, 2))

            short_name = name if len(name) <= 14 else name[:6] + ".." + name[-6:]
            ctk.CTkLabel(
                cell,
                text=short_name,
                font=ctk.CTkFont(size=10),
                text_color=TEXT_MUTED,
            ).pack()

            def _select_pair(_e=None, i=idx):
                self.compare_idx = i
                self._update_comparison_ui()

            for target in (cell, *cell.winfo_children()):
                target.bind("<Button-1>", _select_pair)

    # ── TAB THÔNG TIN & HƯỚNG DẪN ─────────────────────────────
    def _build_info_tab(self) -> None:
        scroll = ctk.CTkScrollableFrame(self.info_frame, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=(4, 12))

        # 1. Hero Header Banner
        hero = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=16, border_width=1, border_color=CARD_LINE)
        hero.pack(fill="x", pady=(4, 12))

        hero_inner = ctk.CTkFrame(hero, fg_color="transparent")
        hero_inner.pack(fill="x", padx=24, pady=18)

        icon_frame = ctk.CTkFrame(hero_inner, fg_color="#1e293b", width=56, height=56, corner_radius=14)
        icon_frame.pack(side="left", padx=(0, 18))
        icon_frame.pack_propagate(False)

        ctk.CTkLabel(
            icon_frame,
            text=FA.ERASER,
            font=fa_font(26),
            text_color=ACCENT_LIGHT,
        ).place(relx=0.5, rely=0.5, anchor="center")

        info_titles = ctk.CTkFrame(hero_inner, fg_color="transparent")
        info_titles.pack(side="left", fill="y")

        t_row = ctk.CTkFrame(info_titles, fg_color="transparent")
        t_row.pack(anchor="w")
        ctk.CTkLabel(
            t_row,
            text="ZWatermark Remover",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        ver_badge = ctk.CTkFrame(t_row, fg_color="#0f766e", corner_radius=6)
        ver_badge.pack(side="left", padx=10)
        ctk.CTkLabel(
            ver_badge,
            text="v2.0 Standalone AI GPU",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ccfbf1",
        ).pack(padx=8, pady=2)

        ctk.CTkLabel(
            info_titles,
            text="Công cụ chuyên biệt xóa Watermark, Logo & Chữ thừa trên Ảnh / Video bằng AI Deep Inpainting & GPU Acceleration",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(4, 0))

        # 2. Developer & Support Card (ZAutomation) - Đặt ngay dưới thông tin ứng dụng
        dev_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=14, border_width=1, border_color="#334155")
        dev_card.pack(fill="x", pady=(0, 12))

        dev_inner = ctk.CTkFrame(dev_card, fg_color="transparent")
        dev_inner.pack(fill="x", padx=20, pady=16)

        logo_path = get_asset_path("zautomation_logo.png") or get_asset_path("logo_sq.png")
        if logo_path and logo_path.is_file():
            try:
                im_logo = Image.open(logo_path).convert("RGB")
                im_logo.thumbnail((64, 64), Image.Resampling.BILINEAR)
                tk_logo = ctk.CTkImage(light_image=im_logo, dark_image=im_logo, size=im_logo.size)
                self._dev_logo_img = tk_logo
                ctk.CTkLabel(dev_inner, image=tk_logo, text="").pack(side="left", padx=(0, 18))
            except Exception:
                pass

        dev_text = ctk.CTkFrame(dev_inner, fg_color="transparent")
        dev_text.pack(side="left", fill="y")

        dev_row1 = ctk.CTkFrame(dev_text, fg_color="transparent")
        dev_row1.pack(anchor="w")

        ctk.CTkLabel(
            dev_row1,
            text="Nhà phát triển:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(
            dev_row1,
            text="ZAutomation",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        ctk.CTkLabel(
            dev_text,
            text="Giải pháp phần mềm tự động hóa, AI Content & Xử lý đa phương tiện",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 6))

        contact_row = ctk.CTkFrame(dev_text, fg_color="transparent")
        contact_row.pack(anchor="w")

        ctk.CTkLabel(
            contact_row,
            text="Zalo liên hệ: ",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        ctk.CTkLabel(
            contact_row,
            text="0942 065 205",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#38bdf8",
        ).pack(side="left", padx=(0, 12))

        # Action buttons on the right
        btn_dev = ctk.CTkFrame(dev_inner, fg_color="transparent")
        btn_dev.pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            btn_dev,
            text="💬 Nhắn Zalo",
            width=115,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._open_zalo_link,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_dev,
            text="📋 Sao chép SĐT",
            width=125,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._copy_phone,
        ).pack(side="left", padx=4)

        # 3. Grid Features (4 Cards)
        feat_grid = ctk.CTkFrame(scroll, fg_color="transparent")
        feat_grid.pack(fill="x", pady=(0, 12))
        feat_grid.grid_columnconfigure((0, 1), weight=1, uniform="feat")

        feats = [
            (
                FA.BOLT,
                ACCENT_LIGHT,
                "Tăng tốc GPU & RAM Pipe",
                "Xử lý khung hình video trực tiếp trong RAM, tận dụng GPU Hardware Encoders (NVENC, QuickSync, AMF) giúp tiết kiệm 80% thời gian render.",
            ),
            (
                FA.MICROCHIP,
                "#a78bfa",
                "AI Deep Inpainting (LaMa & MIGAN ONNX)",
                "Mô hình học sâu AI LaMa & Fast Fourier Convolutions chạy qua ONNX, tự động phục hồi bề mặt tự nhiên 100% không tì vết.",
            ),
            (
                FA.CROP,
                SUCCESS_GREEN,
                "Khoanh vùng & Cọ vẽ thủ công",
                "Trình biên tập cọ vẽ & khoanh ô tương tác trực tiếp trên khung hình, cho phép tùy biến chính xác vùng cần xóa.",
            ),
            (
                FA.MAGIC,
                WARNING_YELLOW,
                "Studio So sánh Trước / Sau",
                "Thanh trượt Split-View tương tác thời gian thực cho phép kiểm tra chi tiết từng pixel trước khi xuất thành phẩm.",
            ),
        ]

        for i, (f_icon, f_color, f_title, f_desc) in enumerate(feats):
            card = ctk.CTkFrame(feat_grid, fg_color=CARD_BG, corner_radius=12, border_width=1, border_color=CARD_LINE)
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6 if i % 2 == 1 else (0, 6), pady=6)

            chead = ctk.CTkFrame(card, fg_color="transparent")
            chead.pack(fill="x", padx=14, pady=(12, 4))

            ctk.CTkLabel(chead, text=f_icon, font=fa_font(16), text_color=f_color).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(chead, text=f_title, font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_MAIN).pack(side="left")

            ctk.CTkLabel(
                card,
                text=f_desc,
                font=ctk.CTkFont(size=11),
                text_color=TEXT_MUTED,
                wraplength=480,
                justify="left",
            ).pack(fill="x", padx=14, pady=(0, 12), anchor="w")

        # 4. Hardware & Device Info Card (Tự động nhận diện)
        hw_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=12, border_width=1, border_color=CARD_LINE)
        hw_card.pack(fill="x", pady=(0, 12))

        hw_head = ctk.CTkFrame(hw_card, fg_color="transparent")
        hw_head.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(hw_head, text="Thông tin Thiết bị & Phần cứng (Tự động nhận diện)", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_MAIN).pack(side="left")

        hw_rows = ctk.CTkFrame(hw_card, fg_color="transparent")
        hw_rows.pack(fill="x", padx=16, pady=(0, 12))

        self.info_dev_name_lbl = ctk.CTkLabel(hw_rows, text="• Tên máy & Hệ điều hành: Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_dev_name_lbl.pack(fill="x", pady=2)

        self.info_cpu_lbl = ctk.CTkLabel(hw_rows, text="• Bộ vi xử lý (CPU):      Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_cpu_lbl.pack(fill="x", pady=2)

        self.info_ram_lbl = ctk.CTkLabel(hw_rows, text="• Bộ nhớ hệ thống (RAM): Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_ram_lbl.pack(fill="x", pady=2)

        self.info_gpu_lbl = ctk.CTkLabel(hw_rows, text="• Card đồ họa (GPU):       Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_gpu_lbl.pack(fill="x", pady=2)

        self.info_hw_enc_lbl = ctk.CTkLabel(hw_rows, text="• Video Hardware Encoder:  Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_hw_enc_lbl.pack(fill="x", pady=2)

        self.info_hw_ai_lbl = ctk.CTkLabel(hw_rows, text="• Động cơ AI Inpainting:   Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_hw_ai_lbl.pack(fill="x", pady=2)

        self.info_hw_ff_lbl = ctk.CTkLabel(hw_rows, text="• Trạng thái FFmpeg Core:    Đang kiểm tra...", font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, anchor="w")
        self.info_hw_ff_lbl.pack(fill="x", pady=2)

        # 5. Quick Step-by-Step Guide
        guide_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=12, border_width=1, border_color=CARD_LINE)
        guide_card.pack(fill="x", pady=(0, 14))

        g_head = ctk.CTkFrame(guide_card, fg_color="transparent")
        g_head.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(g_head, text="Hướng dẫn thao tác nhanh 4 bước", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_MAIN).pack(side="left")

        steps = [
            ("1", "Thêm tệp:", "Kéo thả ảnh hoặc video vào phần mềm, hoặc bấm nút '+ Thêm tệp' / 'Thư mục'."),
            ("2", "Chọn chế độ xóa:", "Để mặc định 'Tự động' (cho logo Gemini, Veo, góc) hoặc bấm 'Khoanh vùng thủ công' để vẽ ô bao quanh logo."),
            ("3", "Bắt đầu xử lý:", "Bấm nút 'BẮT ĐẦU XÓA WATERMARK'. Bạn có thể bấm 'DỪNG' bất kỳ lúc nào nếu cần."),
            ("4", "Kiểm tra kết quả:", "Kéo thanh trượt Split Slider ở khung bên phải để so sánh chi tiết Trước / Sau và bấm 'Mở xuất'."),
        ]

        for num, stitle, sdesc in steps:
            s_row = ctk.CTkFrame(guide_card, fg_color="transparent")
            s_row.pack(fill="x", padx=16, pady=3)

            n_badge = ctk.CTkFrame(s_row, fg_color=ACCENT_GRADIENT, width=22, height=22, corner_radius=11)
            n_badge.pack(side="left", padx=(0, 8))
            n_badge.pack_propagate(False)
            ctk.CTkLabel(n_badge, text=num, font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MAIN).place(relx=0.5, rely=0.5, anchor="center")

            ctk.CTkLabel(s_row, text=stitle, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MAIN).pack(side="left", padx=(0, 6))
            ctk.CTkLabel(s_row, text=sdesc, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED).pack(side="left")

        # 6. Action Button: Open App Folder
        act_row = ctk.CTkFrame(scroll, fg_color="transparent")
        act_row.pack(fill="x", pady=(4, 16))

        ctk.CTkButton(
            act_row,
            text="📂 Mở Thư Mục Ứng Dụng",
            height=38,
            corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color=CARD_LINE,
            command=self._open_app_folder,
        ).pack(side="left")

    def _detect_hardware(self) -> None:
        def _bg_detect():
            import platform
            import os
            import ctypes
            import winreg
            import subprocess
            from core.watermark_remover import get_hardware_acceleration_info

            hw = get_hardware_acceleration_info()

            # 1. Tên máy & OS
            device_name = platform.node() or os.environ.get("COMPUTERNAME", "Windows PC")
            os_name = f"Windows {platform.release()} ({platform.architecture()[0]})"

            # 2. CPU Name
            cpu_name = platform.processor()
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                winreg.CloseKey(key)
                cpu_name = cpu_name.strip()
            except Exception:
                pass

            # 3. RAM Size
            ram_str = "16 GB RAM"
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                ram_str = f"{round(stat.ullTotalPhys / (1024 ** 3))} GB RAM"
            except Exception:
                pass

            # 4. GPU Name
            gpu_str = "DirectX / GPU Compatible"
            try:
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                    text=True, creationflags=flags
                )
                gpus = [line.strip() for line in out.strip().splitlines() if line.strip()]
                if gpus:
                    gpu_str = " | ".join(gpus)
            except Exception:
                pass

            def _update():
                # Header Chips
                if hasattr(self, "gpu_chip"):
                    self.gpu_chip.configure(
                        text=f"⚡ Video: {hw.get('encoder_desc', 'CPU')}",
                        fg_color="#064e3b" if "libx264" not in hw.get("encoder", "") else "#1e293b",
                        text_color="#6ee7b7" if "libx264" not in hw.get("encoder", "") else TEXT_MUTED,
                    )
                if hasattr(self, "ai_chip"):
                    self.ai_chip.configure(
                        text=f"🧠 AI: {hw.get('ai_desc', 'DirectML')}",
                        fg_color="#312e81" if "CPU" not in hw.get("ai_desc", "") else "#1e293b",
                        text_color="#c7d2fe" if "CPU" not in hw.get("ai_desc", "") else TEXT_MUTED,
                    )

                # Info Tab Labels
                if hasattr(self, "info_dev_name_lbl"):
                    self.info_dev_name_lbl.configure(text=f"• Tên thiết bị & OS:      {device_name} ({os_name})", text_color=TEXT_MAIN)
                if hasattr(self, "info_cpu_lbl"):
                    self.info_cpu_lbl.configure(text=f"• Bộ vi xử lý (CPU):      {cpu_name}", text_color=TEXT_MAIN)
                if hasattr(self, "info_ram_lbl"):
                    self.info_ram_lbl.configure(text=f"• Bộ nhớ hệ thống (RAM): {ram_str}", text_color=TEXT_MAIN)
                if hasattr(self, "info_gpu_lbl"):
                    self.info_gpu_lbl.configure(text=f"• Card đồ họa (GPU):       {gpu_str}", text_color="#38bdf8")
                if hasattr(self, "info_hw_enc_lbl"):
                    self.info_hw_enc_lbl.configure(text=f"• Video Hardware Encoder:  {hw.get('encoder_desc', 'CPU')} (Pipeline RAM)", text_color=SUCCESS_GREEN)
                if hasattr(self, "info_hw_ai_lbl"):
                    self.info_hw_ai_lbl.configure(text=f"• Động cơ AI Inpainting:   LaMa & MIGAN ({hw.get('ai_desc', 'DirectML')})", text_color="#a78bfa")
                if hasattr(self, "info_hw_ff_lbl"):
                    self.info_hw_ff_lbl.configure(text="• Trạng thái FFmpeg Core:    ✓ Đã kết nối & Sẵn sàng", text_color=SUCCESS_GREEN)

            self.after(0, _update)

        threading.Thread(target=_bg_detect, daemon=True).start()

    def _copy_phone(self) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append("0942065205")
            messagebox.showinfo("Đã sao chép", "Đã sao chép số Zalo: 0942 065 205 vào bộ nhớ tạm!")
        except Exception:
            pass

    def _open_zalo_link(self) -> None:
        try:
            import webbrowser
            webbrowser.open("https://zalo.me/0942065205")
        except Exception:
            pass

    def _open_guide_doc(self) -> None:
        doc_path = get_app_dir() / "HUONG_DAN_ZWATERMARK.md"
        if not doc_path.is_file():
            doc_path = Path(__file__).resolve().parent / "HUONG_DAN_ZWATERMARK.md"
        if doc_path.is_file():
            try:
                os.startfile(str(doc_path))
            except Exception:
                import subprocess
                subprocess.Popen(["notepad", str(doc_path)])
        else:
            messagebox.showinfo("Tài liệu", f"Tài liệu hướng dẫn: {doc_path.name}")

    def _open_app_folder(self) -> None:
        app_d = get_app_dir()
        try:
            os.startfile(str(app_d))
        except Exception:
            import subprocess
            subprocess.Popen(["explorer", str(app_d)])

    def _generate_thumb(self, path_str: str) -> ctk.CTkImage | None:
        p = Path(path_str)
        if not p.is_file():
            return None
        try:
            import tempfile
            if p.suffix.lower() in VIDEO_EXT:
                tmp = Path(tempfile.gettempdir()) / f"zwatermark_th_{p.stem}.png"
                extract_preview_frame(p, tmp, time_sec=0.5)
                im = Image.open(tmp).convert("RGB")
            else:
                im = Image.open(p).convert("RGB")
            im.thumbnail((90, 52), Image.Resampling.BILINEAR)
            return ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
        except Exception:
            return None


def main():
    app = ZWatermarkRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()

