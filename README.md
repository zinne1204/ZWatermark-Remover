# 🧼 ZWatermark Remover AI — GPU & Deep Inpainting

<p align="center">
  <img src="assets/logo.png" alt="ZWatermark Remover Logo" width="160"/>
</p>

<p align="center">
  <b>Công cụ Windows Desktop mã nguồn mở chuyên biệt để xóa Watermark, Logo, Phụ đề & Chữ chìm trên cả Hình ảnh và Video bằng AI Deep Inpainting & Tăng tốc GPU DirectML/CUDA.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-0078D6.svg" alt="Windows"/>
  <img src="https://img.shields.io/badge/GPU%20Acceleration-DirectML%20%7C%20CUDA%20%7C%20NVENC-success.svg" alt="GPU Support"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

---

## ✨ Tính Năng Nổi Bật (Key Features)

- 🤖 **✨ Tự động Toàn diện (Smart Auto):** Tự động nhận diện và xóa sạch logo ở 4 góc màn hình, vệt sáng AI (Gemini, Veo, TikTok, CapCut...) mà không làm mất nét chủ thể chính.
- 📏 **Thước Kéo Thẳng (Straight Line Brush):** Kéo 1 đường thẳng tắp theo thước để gạch xoá dòng chữ watermark ngang hoặc chéo siêu tốc và chuẩn xác.
- 🖌️ **Studio Khoanh Vùng & Cọ Vẽ Tự Do:** Tùy biến kích cỡ cọ vẽ (`6px` - `80px`), kéo ô chữ nhật (`Box ROI`), cọ tẩy (`Eraser`).
- ↩️ **Hoàn tác Đa tầng (Undo / Ctrl+Z):** Lưu lịch sử tới 30 bước vẽ, dễ dàng khôi phục nét vẽ trước đó nếu thao tác nhầm.
- ⚡ **Tăng tốc GPU DirectML / CUDA:** Tận dụng tối đa sức mạnh card đồ họa (NVIDIA, AMD Radeon, Intel Arc/Iris Xe) để xử lý video trực tiếp trên RAM qua pipeline FFmpeg không nén.
- 🔍 **Studio So Sánh Trước / Sau (Split View Slider):** Kéo thanh trượt để so sánh trực quan từng chi tiết ảnh gốc và ảnh đã xóa watermark trước khi xuất.
- 🔒 **100% Offline & Bảo Mật:** Xử lý hoàn toàn cục bộ trên máy tính của bạn, không upload ảnh/video lên máy chủ bên ngoài.

---

## 📁 Định Dạng Hỗ Trợ (Supported Formats)

| Loại Tệp | Các định dạng hỗ trợ |
| :--- | :--- |
| **Video** | `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi` |
| **Hình ảnh** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` |

---

## 🚀 Cài Đặt & Khởi Chạy (Quick Start)

### 1. Yêu cầu hệ thống:
* **Hệ điều hành:** Windows 10 / Windows 11 (64-bit).
* **Python:** 3.9 trở lên (Khuyên dùng Python 3.11 hoặc 3.12).
* **FFmpeg:** Tải bản FFmpeg Essentials và thêm vào PATH hệ thống (hoặc đặt `ffmpeg.exe` vào thư mục `assets/bin/`).

### 2. Cài đặt thư viện:
```bash
git clone https://github.com/your-username/zwatermark-remover.git
cd zwatermark-remover
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng:
```bash
python app.py
```
*(Hoặc click đúp chuột vào file `run.bat`)*

---

## 🛠️ Hướng Dẫn Đóng Gói .EXE (Build Standalone Executable)

Dự án hỗ trợ sẵn 2 kịch bản biên dịch C++ tối ưu bằng **Nuitka**:

1. **Đóng gói Thư mục Standalone (Khuyên dùng, chạy nhanh nhất & không bị Antivirus chặn):**
   ```bash
   build-standalone.bat
   ```
   *File `.exe` sẽ được tạo tại:* `dist_nuitka\ZWatermarkRemover\ZWatermarkRemover.exe`

2. **Đóng gói Onefile (1 file `.exe` duy nhất):**
   ```bash
   build-onefile.bat
   ```
   *File `.exe` sẽ được tạo tại:* `dist_onefile\ZWatermarkRemover.exe`

---

## 🎮 Phím Tắt Tiện Lợi (Keyboard Shortcuts)

* `Ctrl + Z`: Hoàn tác (Undo) nét vẽ cọ / thước kẻ trong Studio.
* `Drag & Drop`: Kéo thả trực tiếp hàng loạt ảnh hoặc video vào phần mềm.
* `Split Slider`: Kéo thanh trượt chuột trái qua lại trên khung Studio để so sánh.

---

## 📄 Bản Quyền & Giấy Phép (License)

Phần mềm được phát hành theo giấy phép mã nguồn mở **[MIT License](LICENSE)**. Bạn hoàn toàn có quyền sử dụng miễn phí cho mục đích cá nhân và thương mại.
