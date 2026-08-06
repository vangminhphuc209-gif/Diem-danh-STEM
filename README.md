# 🎓 Hệ Thống Điểm Danh Thông Minh — STEM Project v4.0

> **Điểm danh tự động** bằng QR Code + Nhận diện khuôn mặt + **Kiểm tra trang phục** theo quy định nhà trường, chạy realtime qua camera.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-2.x-orange?logo=tensorflow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Raspberry%20Pi-lightgrey)

---

## 📋 Mục Lục

- [Tổng quan hệ thống](#-tổng-quan-hệ-thống)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc Hybrid AI](#-kiến-trúc-hybrid-ai--opencv)
- [Quy định trang phục theo ngày](#-quy-định-trang-phục-theo-ngày)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Pipeline AI trang phục](#-pipeline-ai-trang-phục-bước-từng-bước)
- [Phím tắt](#-phím-tắt)
- [Cấu hình](#-cấu-hình)
- [Dashboard & Báo cáo](#-dashboard--báo-cáo)

---

## 🌟 Tổng Quan Hệ Thống

```
┌────────────────────────────────────────────────────────────────────┐
│                        VIDEO STREAM (Camera)                       │
└───────────────────────────────┬────────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │         HOG Person Detector         │
              │    (phát hiện thân người → bbox)    │
              └──────┬──────────────────────┬───────┘
                     │                      │
        ┌────────────▼────────┐   ┌─────────▼─────────────┐
        │   Face Recognition  │   │  Uniform Checker       │
        │  (Haar → dlib/FR)   │   │  (AI + Color Hybrid)   │
        └────────────┬────────┘   └─────────┬─────────────┘
                     │                      │
        ┌────────────▼────────┐   ┌─────────▼─────────────┐
        │  Điểm danh QR Code  │   │  Kết quả trang phục    │
        │  Google Sheets Sync │   │  OK / FAIL / UNCLEAR   │
        │  SQLite Offline Buf │   │  (UNCLEAR → chụp ảnh   │
        └─────────────────────┘   │   → upload cloud)      │
                                  └────────────────────────┘
```

---

## ✨ Tính Năng Chính

### 📌 Điểm Danh
| Tính năng | Mô tả |
|-----------|-------|
| **QR Code** | Quét thẻ QR cá nhân để điểm danh |
| **Nhận diện khuôn mặt** | Haar Cascade (nhanh) + face_recognition (chính xác) |
| **Phát hiện người lạ** | Tự động chụp + ghi Sheets khi không nhận ra |
| **Offline buffer** | SQLite lưu dự phòng khi mất mạng, tự đồng bộ khi có mạng |
| **Google Sheets** | Ghi điểm danh realtime lên cloud |
| **MJPEG Stream** | Xem camera qua browser tại `http://<IP>:8080` |

### 👕 Kiểm Tra Trang Phục (v4.0 — Mới)
| Tính năng | Mô tả |
|-----------|-------|
| **HOG Person Detector** | Phát hiện thân người chính xác hơn face-only |
| **AI TFLite (Thứ 2)** | Nhận diện trang phục dân tộc vs. không phải dân tộc |
| **HSV Color Analysis (Thứ 3, 5, 6)** | Phân tích màu áo trắng / xanh đoàn + quần tối |
| **Hybrid Pipeline** | AI + Color = nhanh, chính xác, ít tốn tài nguyên |
| **UNCLEAR → Upload** | Trang phục không xác định được → chụp ảnh → gửi cloud |
| **HSV Calibrator** | Tool GUI chỉnh ngưỡng màu theo camera thực tế |

---

## 🏗 Kiến Trúc Hybrid AI + OpenCV

Hệ thống **không dùng AI cho mọi thứ** — đây là chiến lược tối ưu:

| Ngày | Phương pháp | Lý do |
|------|-------------|-------|
| **Thứ 2** | 🤖 TFLite AI (MobileNetV2) | Trang phục dân tộc có đặc trưng hình ảnh phức tạp |
| **Thứ 3 & 5** | 🎨 HSV Color Analyzer | Áo trắng = HSV Saturation thấp, Value cao → đơn giản, nhanh |
| **Thứ 4** | ⏭ Bỏ qua | "Có cổ" cần pose estimation — chưa triển khai |
| **Thứ 6** | 🎨 HSV Color Analyzer | Áo đoàn xanh = HSV Hue ∈ [95–135] → đặc trưng rõ |
| **Thứ 7 & CN** | ⏭ Bỏ qua | Không yêu cầu |

> **Kết quả:** Giảm ~60–70% tải cho model AI so với dùng AI cho tất cả ngày.

---

## 📅 Quy Định Trang Phục Theo Ngày

```
Thứ 2  │ 🥻 Trang phục dân tộc (bản cách điệu được chấp nhận)
Thứ 3  │ 👔 Áo sơ mi TRẮNG + Quần TỐI màu
Thứ 4  │ 👕 Tự do — phải có cổ (sơ mi / polo)
Thứ 5  │ 👔 Áo sơ mi TRẮNG + Quần TỐI màu
Thứ 6  │ 🔵 Áo Đoàn TNCS (xanh dương đặc trưng) + Quần TỐI màu
Thứ 7  │ 🆓 Tự do — không yêu cầu
CN     │ 🆓 Tự do — không yêu cầu
```

### Trạng thái trả về:
| Status | Màu hiển thị | Ý nghĩa |
|--------|-------------|---------|
| `OK` | 🟢 Xanh lá | Đúng trang phục |
| `FAIL` | 🔴 Đỏ | Sai trang phục |
| `UNCLEAR` | 🟠 Cam | Không xác định → **Camera chụp → Upload cloud** |
| `SKIP` | 🔵 Xanh dương | Ngày không yêu cầu |

---

## 📁 Cấu Trúc Thư Mục

```
Diem-danh-STEM/
│
├── 🚀 CHẠY CHÍNH
│   ├── main.py                     # Chương trình chính (điểm danh + trang phục)
│   ├── config.py                   # Toàn bộ cấu hình hệ thống
│   └── requirements.txt            # Thư viện Python
│
├── 🤖 MODULE AI TRANG PHỤC (v4.0)
│   ├── person_detector.py          # HOG person detector + face fallback + NMS
│   ├── uniform_checker.py          # Bộ kiểm tra tổng hợp (router theo ngày)
│   ├── uniform_color_analyzer.py   # Phân tích màu sắc HSV (Thứ 3,5,6)
│   ├── uniform_trainer.py          # Training model TFLite MobileNetV2
│   └── uniform_run.py              # Chạy kiểm tra trang phục độc lập
│
├── 📊 BƯỚC 1 — CHUẨN BỊ DỮ LIỆU
│   ├── step1_dataset_audit.py      # Kiểm kê & báo cáo dataset
│   ├── step1_data_prep_v2.py       # Tạo dataset 2-nhãn + augmentation
│   └── step1_hsv_calibrator.py     # GUI calibrate ngưỡng màu theo camera
│
├── 👤 MODULE NHẬN DIỆN KHUÔN MẶT
│   ├── face_engine.py              # Face Engine v3 (Haar + face_recognition)
│   └── utils.py                    # Hàm hỗ trợ (HUD, mạng, âm thanh...)
│
├── 🔧 THIẾT LẬP
│   ├── setup_sheets.py             # Khởi tạo Google Sheets (chạy 1 lần)
│   ├── init_db.py                  # Khởi tạo SQLite (chạy 1 lần)
│   ├── generate_qr.py              # Tạo thẻ QR (--sheet, --pdf, --preview)
│   └── api_server.py               # API REST server
│
├── 📈 BÁO CÁO
│   └── dashboard.py                # Báo cáo HTML với biểu đồ Chart.js
│
└── 📂 data/
    ├── database.db                 # SQLite offline buffer
    ├── stuface/                    # Ảnh khuôn mặt học sinh (HS001.jpg ...)
    ├── violations/                 # Ảnh vi phạm trang phục + người lạ
    ├── uniform_picdemo/            # Dataset ảnh trang phục gốc
    │   ├── trang_phuc_dan_toc/     # ~211 ảnh (24 dân tộc nam + 25 dân tộc nữ)
    │   ├── ao_trang/               # Ảnh áo trắng (thêm vào để cải thiện model)
    │   ├── ao_doan/                # Ảnh áo đoàn (thêm vào để cải thiện model)
    │   └── other/                  # Ảnh trang phục khác (negative samples)
    ├── uniform_dataset_v2/         # Dataset đã xử lý (train/val, 2 nhãn)
    ├── uniform_model_v2.tflite     # Model AI trang phục (sau khi training)
    ├── uniform_labels_v2.json      # Nhãn model
    ├── hsv_config.json             # Ngưỡng HSV đã calibrate theo camera
    └── audit_report.txt            # Báo cáo kiểm kê dataset
```

---

## 🔧 Cài Đặt

### Yêu cầu hệ thống
- Python **3.10+**
- Camera USB hoặc IP Camera
- RAM tối thiểu: **4GB** (khuyến nghị 8GB)
- Hệ điều hành: Windows 10/11 hoặc Raspberry Pi OS

### 1. Clone dự án
```bash
git clone <repo-url>
cd Diem-danh-STEM
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt

# Thêm thư viện cho AI trang phục
pip install tensorflow scikit-learn

# Windows: cần cmake trước khi cài dlib
pip install cmake
pip install dlib face_recognition
```

> **Lưu ý Raspberry Pi:** Dùng `tflite-runtime` thay cho `tensorflow` để nhẹ hơn:
> ```bash
> pip install tflite-runtime
> ```

### 3. Cấu hình Google Sheets
```bash
# Đặt file credentials vào thư mục gốc
# Tên file: reflecting-site-494013-b3-9978cb2582bd.json

python setup_sheets.py    # Tạo cấu trúc sheet (chạy 1 lần)
python init_db.py         # Tạo SQLite database (chạy 1 lần)
```

### 4. Thêm ảnh học sinh
```bash
# Đặt ảnh khuôn mặt vào data/stuface/
# Tên file = Mã học sinh (ví dụ: HS001.jpg, HS002.jpg)
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Chạy hệ thống chính
```bash
python main.py
```

### Pipeline AI trang phục — Chạy từng bước

#### Bước 1A: Kiểm kê dataset
```bash
python step1_dataset_audit.py
# Xuất báo cáo: data/audit_report.txt
```

#### Bước 1B: Chuẩn bị dataset
```bash
python step1_data_prep_v2.py
# Tạo: data/uniform_dataset_v2/ (train + val, 2 nhãn)

# Tùy chọn:
python step1_data_prep_v2.py --aug-factor 20    # Augment nhiều hơn
python step1_data_prep_v2.py --no-aug           # Không augment (dùng ảnh gốc)
```

#### Bước 1C: Calibrate màu HSV (quan trọng!)
```bash
python step1_hsv_calibrator.py
# Mở camera → mặc áo trắng / xanh đứng trước → chỉnh slider → nhấn S để lưu
# Kết quả: data/hsv_config.json (tự động load khi chạy)

# Camera khác:
python step1_hsv_calibrator.py --camera 1
```

#### Bước 2: Huấn luyện model AI
```bash
python uniform_trainer.py
# Output: data/uniform_model_v2.tflite + data/uniform_labels_v2.json
```

#### Bước 3: Test kết quả
```bash
python uniform_checker.py --test           # Test qua webcam
python uniform_checker.py --image <path>   # Test trên ảnh tĩnh
python person_detector.py                  # Test HOG detector riêng lẻ
python uniform_color_analyzer.py           # Test color analyzer (có D=debug mask)
```

---

## ⌨️ Phím Tắt (Khi Đang Chạy)

| Phím | Chức năng |
|------|-----------|
| `Q` / `ESC` | Thoát (tự ghi tổng kết cuối ngày) |
| `S` | Xem thống kê: có mặt / vắng / vi phạm trang phục / người lạ |
| `R` | Reload danh sách học sinh + khuôn mặt (hot-reload) |
| `P` | Tạm dừng / tiếp tục nhận diện khuôn mặt |
| `A` | Thêm khuôn mặt mới trực tiếp từ camera |

**Khi dùng `step1_hsv_calibrator.py`:**

| Phím | Chức năng |
|------|-----------|
| `1` | Chế độ Trắng (Thứ 3, 5) |
| `2` | Chế độ Xanh Dương (Thứ 6) |
| `3` | Chế độ Tối màu (Quần) |
| `D` | Bật/tắt debug — hiện giá trị HSV tại điểm con trỏ |
| `S` | Lưu cấu hình vào `data/hsv_config.json` |
| `R` | Reset về giá trị mặc định |
| `Q` | Thoát |

**Khi dùng `person_detector.py`:**

| Phím | Chức năng |
|------|-----------|
| `Z` | Bật/tắt hiển thị vùng áo / vùng quần |
| `Q` | Thoát |

---

## ⚙️ Cấu Hình (`config.py`)

### Camera & Mạng
| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `CAMERA_INDEX` | Index webcam (0=mặc định, hoặc URL RTSP/IP) | `0` |
| `STREAM_ENABLE` | Bật MJPEG stream qua browser | `True` |
| `STREAM_PORT` | Cổng HTTP stream | `8080` |
| `STREAM_FPS` | FPS stream tối đa | `15` |

### Nhận diện khuôn mặt
| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `FACE_TOLERANCE` | Ngưỡng khớp khuôn mặt (0.4=chặt, 0.6=thoáng) | `0.5` |
| `HAAR_MIN_NEIGHBORS` | Độ nhạy Haar (thấp=nhạy hơn) | `4` |
| `FACE_SKIP_FRAMES` | Chạy face_recognition mỗi N frame | `5` |
| `CAPTURE_COOLDOWN` | Giây giữa 2 lần chụp vi phạm | `30` |

### HUD & Hiển thị
| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `SHOW_CLOCK` | Đồng hồ trên camera | `True` |
| `SHOW_STATS` | Thống kê trên camera | `True` |

### Trang phục AI (`uniform_checker.py`)
| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `CONFIDENCE_THRESHOLD` | Ngưỡng tin cậy TFLite (thấp hơn → UNCLEAR) | `0.55` |
| `IMAGE_SIZE` | Kích thước ảnh đầu vào model | `(224, 224)` |

### Màu sắc HSV (có thể calibrate qua `step1_hsv_calibrator.py`)
| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `WHITE_S_MAX` | Saturation tối đa để là "trắng" | `60` |
| `WHITE_V_MIN` | Độ sáng tối thiểu để là "trắng" | `160` |
| `BLUE_H_MIN/MAX` | Dải Hue của áo đoàn xanh | `95–135` |
| `DARK_V_MAX` | Độ sáng tối đa để coi là "tối màu" | `100` |

---

## 📊 Dashboard & Báo Cáo

```bash
python dashboard.py --open             # Báo cáo hôm nay (mở trình duyệt)
python dashboard.py --date 06/08/2026  # Báo cáo ngày cụ thể
```

Dashboard bao gồm:
- 📈 Biểu đồ điểm danh theo giờ
- 📋 Danh sách có mặt / vắng
- 👕 Thống kê vi phạm trang phục
- 🚨 Danh sách trang phục UNCLEAR (cần người kiểm tra)

---

## 🎨 Màu Sắc Trên Camera

### Khuôn mặt
| Màu | Ý nghĩa |
|-----|---------|
| 🟢 **Xanh lá** | Đã quét QR + nhận diện OK |
| 🟡 **Vàng cam** | Nhận ra nhưng CHƯA quét thẻ |
| 🔴 **Đỏ** | Người lạ / không trong danh sách |
| ⬜ **Xám** | Đang scan nhanh (Haar), chưa xác định |

### Trang phục
| Màu | Ý nghĩa |
|-----|---------|
| 🟢 **Xanh lá** | Đúng trang phục (`OK`) |
| 🔴 **Đỏ** | Sai trang phục (`FAIL`) |
| 🟠 **Cam** | Không xác định (`UNCLEAR`) → chụp ảnh |
| 🔵 **Xanh dương** | Ngày không yêu cầu (`SKIP`) |

---

## 🧠 Chi Tiết Kỹ Thuật — Module AI Trang Phục

### Model AI (Thứ 2 — Trang phục dân tộc)
- **Kiến trúc:** MobileNetV2 (Transfer Learning từ ImageNet)
- **2 nhãn:** `trang_phuc_dan_toc` | `other`
- **Kết quả `other`:** → `UNCLEAR` → camera chụp → upload cloud
- **Dataset:** ~211 ảnh gốc (24 dân tộc nam + 25 dân tộc nữ), augmented lên 672+ ảnh
- **Export:** TFLite INT8 quantization (~4–6 MB)
- **Inference time:** ~30ms/frame (CPU), ~10ms (GPU/Edge TPU)

### HOG Person Detector
- **Detector:** `cv2.HOGDescriptor` + `HOGDescriptor_getDefaultPeopleDetector()`
- **Fallback:** Haar Cascade face → ước tính thân người (×1.8 rộng, ×7 cao)
- **NMS:** Non-Maximum Suppression loại hộp chồng nhau (threshold 0.65)
- **Tốc độ:** Frame scale 60% trước khi detect để tăng FPS

### Color Analyzer (Thứ 3, 5, 6)
```
Vùng ÁO   : 12% → 52% chiều cao body_box
Vùng QUẦN : 52% → 90% chiều cao body_box

Áo TRẮNG  : HSV Saturation ≤ s_max, Value ≥ v_min
Áo XANH   : HSV Hue ∈ [h_min, h_max], Saturation ≥ s_min
Quần TỐI  : HSV Value ≤ v_max (≥ 45% pixels vùng quần)
```

---

## 📌 Lưu Ý Quan Trọng

> **Dataset `other` còn ít (7 ảnh gốc).** Để model chính xác hơn, hãy thêm ảnh vào:
> ```
> data/uniform_picdemo/other/         ← Ảnh đồng phục thường, áo thường
> data/uniform_picdemo/ao_trang/      ← Thêm ảnh áo trắng
> data/uniform_picdemo/ao_doan/       ← Thêm ảnh áo đoàn
> ```
> Sau đó chạy lại: `python step1_data_prep_v2.py` → `python uniform_trainer.py`

> **Luôn calibrate HSV** trước khi deploy vào phòng học thực tế vì ánh sáng đèn huỳnh quang ảnh hưởng đáng kể đến phân tích màu sắc.

---

## 📦 Thư Viện Python

```
opencv-python >= 4.8.0          # Xử lý ảnh, HOG detector, HSV
opencv-contrib-python >= 4.8.0  # Haar Cascade extra models
pyzbar >= 0.1.9                 # Đọc QR code
gspread >= 6.0.0                # Google Sheets API
oauth2client >= 4.1.3           # Google OAuth
qrcode[pil] >= 7.4.2            # Tạo QR code
Pillow >= 10.0.0                # Xử lý ảnh PIL
face-recognition >= 1.3.0       # Nhận diện khuôn mặt (cần cmake + dlib)
numpy >= 1.24.0                 # Ma trận số học
tensorflow >= 2.13.0            # Training model (hoặc tflite-runtime trên Pi)
scikit-learn >= 1.3.0           # class_weight tính toán cân bằng nhãn
```

---

*STEM Project v4.0 — Hệ thống Điểm danh Thông minh: QR Code + Face Recognition + AI Uniform Checker*
