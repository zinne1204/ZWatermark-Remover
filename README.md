# HƯỚNG DẪN SỬ DỤNG PHẦN MỀM ZWATERMARK REMOVER
**Công cụ chuyên biệt xóa Watermark, Logo & Chữ thừa trên Ảnh & Video bằng AI GPU Inpainting**

---

## 🌟 1. GIỚI THIỆU
**ZWatermark Remover** là phần mềm Windows Desktop độc lập, chuyên sâu cho việc tẩy xóa các loại watermark mờ, logo thương hiệu, chữ bản quyền, subtitle hoặc icon thừa trên cả hình ảnh và video với tốc độ cao nhờ tận dụng tối đa sức mạnh phần cứng GPU.

### ✨ Tính năng nổi bật:
* 🚀 **Chạy độc lập & siêu nhẹ:** Khởi động cực nhanh, không phụ thuộc vào trình dựng video bên ngoài.
* ⚡ **Tăng tốc GPU toàn diện:**
  * **AI Inpainting:** Tích hợp mô hình Deep Learning MIGAN ONNX chạy trực tiếp trên GPU qua **DirectML** (hỗ trợ mọi card đồ họa AMD, Intel, NVIDIA) hoặc **CUDA**.
  * **Video Processing RAM Pipe:** Xử lý từng khung hình video trực tiếp trong bộ nhớ RAM và xuất qua GPU Hardware Encoders (**NVIDIA NVENC, Intel QuickSync, AMD AMF**) giúp tiết kiệm 80% thời gian render.
* 🎯 **2 Chế độ xóa linh hoạt:**
  1. **Tự động nhận diện (AI Auto Detect):** Nhận diện watermark sparkle của Google Gemini, Veo AI và logo ở 4 góc video.
  2. **Khoanh vùng thủ công (Interactive Region Selector):** Kéo chuột vẽ ô bao quanh logo trên khung hình mẫu để xóa chính xác bất kỳ vị trí nào trên ảnh/video hàng loạt.
* 🎚️ **Studio So sánh Trước / Sau (Split View Slider):** Kéo thanh trượt qua lại để kiểm tra chất lượng phục hồi bề mặt từng pixel trước khi sử dụng.
* ⏹ **Nút Dừng (Stop) an toàn:** Dừng tiến trình ngay lập tức khi đang chạy video dài mà không làm treo máy.

---

## 📁 2. ĐỊNH DẠNG TỆP HỖ TRỢ
Phần mềm hỗ trợ kéo thả và xử lý hàng loạt:
* **Video:** `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`
* **Hình ảnh:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

---

## 🚀 3. HƯỚNG DẪN SỬ DỤNG TỪNG BƯỚC

### Bước 1: Thêm ảnh hoặc video
* **Cách 1:** Kéo và thả trực tiếp một hoặc nhiều file ảnh/video vào khung làm việc của phần mềm.
* **Cách 2:** Bấm nút **`+ Thêm tệp`** hoặc **`📁 Thư mục`** ở góc trên danh sách tệp.

### Bước 2: Chọn Chế độ xóa Watermark
1. **⚡ Tự động (Góc & Sparkle AI - Mặc định):**
   * Tự động tìm và xóa logo ở 4 góc hoặc biểu tượng lấp lánh (Gemini, Veo).
2. **📝 Quét Chữ chéo / Stock Watermark toàn màn hình:**
   * Dành riêng cho ảnh/video có **dải chữ mờ chạy chéo khắp ảnh** (như Shutterstock, Getty, iStock, Adobe Stock).
   * AI sẽ tự động phân tích tần số cao và bóc tách nét chữ chéo, sau đó dùng thuật toán **Tiled AI Inpainting** chia ô để phục hồi toàn bộ bức ảnh cực nét.
3. **🎯 Thủ công (Bút vẽ cọ & Khoanh ô):**
   * Bấm nút **`🎯 Bút vẽ & Khoanh ô`**.
   * Chọn công cụ **`🖌️ Bút Cọ Vẽ (Brush)`** (chỉnh cỡ cọ tùy ý) để quẹt trực tiếp lên chữ/logo hoặc chọn **`📦 Kéo Ô (Box)`** để khoanh vùng hình chữ nhật.
   * Bấm **`✓ Dùng vùng / nét vẽ này`**. Tọa độ và mặt nạ sẽ tự động áp dụng chính xác cho toàn bộ danh sách tệp.

### Bước 3: Cấu hình nâng cao (Tùy chọn)
* **AI Inpainting Engine:** Bật công tắc để sử dụng AI Inpainting chuyên sâu (khử mờ sạch sẽ và tái tạo chi tiết nền) hoặc tắt để dùng thuật toán nội suy Telea siêu tốc.
* **Thư mục xuất file:** 
  * Để trống: Tự động lưu file sạch cạnh file gốc với tên `<tên_gốc>_removed_watermark.<đuôi>`.
  * Hoặc bấm **`Chọn`** để chỉ định thư mục lưu tập trung.

### Bước 4: Bắt đầu xử lý
* Bấm nút **`🚀 BẮT ĐẦU XÓA WATERMARK`**.
* Theo dõi tiến độ thời gian thực (FPS, số frame đã xóa, thời gian còn lại) trên thanh tiến độ và khung nhật ký (Live Log).
* Khi cần dừng, bạn có thể bấm nút **`⏹ DỪNG`** bất kỳ lúc nào.

### Bước 5: Kiểm tra kết quả trong Studio So sánh Trước / Sau
* Sau khi xử lý xong, khung **Studio So sánh Trước / Sau** bên phải sẽ tự động hiển thị kết quả.
* **Kéo thanh trượt Split Slider** qua lại để đối chiếu sự khác biệt giữa bản gốc (bên trái) và bản đã xóa watermark (bên phải).
* Bấm vào các ảnh nhỏ trong dải **Thư viện kết quả hoàn thành** ở phía dưới để chuyển đổi nhanh giữa các video/ảnh khác nhau.
* Bấm **`📂 Mở xuất`** để mở thư mục chứa thành phẩm.

---

## 💡 4. MẸO & TỐI ƯU HIỆU NĂNG

> [!TIP]
> **1. Khoanh vùng vừa khít logo:**
> Khi khoanh vùng thủ công, bạn nên vẽ ô bao vừa đủ logo (không cần vẽ quá rộng) để AI inpainting tập trung xử lý đúng vị trí, giúp tái tạo nền tự nhiên nhất và tối ưu tốc độ.

> [!TIP]
> **2. Tận dụng GPU Hardware Acceleration:**
> Phần mềm tự động nhận diện và kích hoạt card đồ họa NVIDIA (NVENC), Intel (QuickSync) hoặc AMD (AMF). Đảm bảo driver card màn hình của bạn đã được cập nhật bản mới nhất để đạt FPS cao nhất.

---

## 📞 5. THÔNG TIN NHÀ PHÁT TRIỂN & HỖ TRỢ KỸ THUẬT
* **Đơn vị phát triển:** **ZAutomation** — *Giải pháp phần mềm tự động hóa & AI đa phương tiện*
* **Hotline / Zalo liên hệ:** **`0942 065 205`**
* **Trực tiếp qua Zalo:** [https://zalo.me/0942065205](https://zalo.me/0942065205)

---

**Chúc bạn có những video và hình ảnh sạch đẹp, chuyên nghiệp cùng ZWatermark Remover!** ✨🎬
