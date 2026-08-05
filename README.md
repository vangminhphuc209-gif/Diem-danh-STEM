# 📷 Hệ Thống Điểm Danh QR Code — STEM Project v3.0

## 🆕 v3.0 — Merge hoàn hảo v1-modified + v2

v3.0 tích hợp TẤT CẢ những gì tốt nhất từ cả hai phiên bản:

### Từ v1-modified (bạn đã tự thêm vào)
| Tính năng | Mô tả |
|-----------|-------|
| **Haar Cascade 2 lớp** | Lớp 1: Haar Cascade phát hiện MỌI khuôn mặt mỗi frame (nhanh). Lớp 2: face_recognition đối chiếu danh tính định kỳ (chính xác) |
| **Nhận diện NGƯỜI LẠ** | Tự động chụp ảnh + ghi Sheets khi phát hiện người không trong danh sách |
| **Sidebar thông tin** | Bảng thông tin đẹp (ID / Tên / Trạng thái / %) bên cạnh mỗi khung mặt |
| **Box 4-góc** | Khung nhận diện mặt dùng 4 góc cắt thay vì hình chữ nhật thô |
| **Status 3 mức** | `ok` (đã quét thẻ) / `warn` (biết nhưng chưa quét) / `stranger` (người lạ) |
| **capture_violation()** | Cooldown chụp ảnh tránh spam file |

### Từ v2 (bug fix + tính năng v2)
| Tính năng | Mô tả |
|-----------|-------|
| **Bug: Camera vòng lặp vô hạn** | Thêm `MAX_CAM_FAILS = 30` |
| **Bug: RAM leak** | Dọn `recently_scan` / `last_results` mỗi 300 frame |
| **Bug: Unicode crash Windows** | ASCII fallback cho `print_log()` |
| **Bug: Mất dữ liệu khi Sheets lỗi** | SQLite offline buffer, đồng bộ khi có mạng |
| **Phím tắt đầy đủ** | S / R / P / A |
| **Tổng kết cuối ngày** | Ghi tự động lên sheet `TongKet` (gồm cả cột Người Lạ) |
| **Trạng thái mạng** | [ONLINE]/[OFFLINE] realtime trên camera |
| **add_face_live()** | Thêm khuôn mặt ngay trong runtime (phím A) |
| **reload()** | Hot-reload không cần tắt chương trình (phím R) |
| **draw_banner / draw_clock / draw_stats** | HUD đẹp hơn trên camera |
| **dashboard.py** | Báo cáo HTML với biểu đồ Chart.js |

---

## Cấu trúc thư mục

```
du an 1/
├── main.py            # Chương trình chính v3.0
├── config.py          # Cấu hình (Haar + v2 settings)
├── utils.py           # Hàm hỗ trợ (HUD, mạng, âm thanh...)
├── face_engine.py     # Face Engine v3 (Haar + face_recognition)
├── setup_sheets.py    # Khởi tạo Google Sheets
├── generate_qr.py     # Tạo thẻ QR (--sheet, --pdf, --preview)
├── dashboard.py       # Báo cáo HTML trực quan
├── init_db.py         # Khởi tạo SQLite
├── test_offline.py    # Kiểm tra camera + QR offline
├── students.csv       # Danh sách học sinh
├── requirements.txt   # Thư viện Python
├── data/
│   ├── database.db    # SQLite offline buffer
│   ├── stuface/       # Ảnh khuôn mặt (HS001.jpg...)
│   └── violations/    # Ảnh vi phạm + người lạ
└── qr_cards/          # Thẻ QR đã tạo
```

---

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
python setup_sheets.py    # Khởi tạo Sheets (chạy 1 lần)
python init_db.py         # Khởi tạo SQLite (chạy 1 lần)
python main.py            # Chạy hệ thống
```

---

## ⌨️ Phím tắt

| Phím | Chức năng |
|------|-----------|
| `Q` / `ESC` | Thoát (tự ghi tổng kết) |
| `S` | Thống kê có mặt / vắng / vi phạm / người lạ |
| `R` | Reload danh sách học sinh + khuôn mặt |
| `P` | Tạm dừng / tiếp tục nhận diện mặt |
| `A` | Thêm khuôn mặt mới từ camera ngay lập tức |

---

## 🎨 Màu sắc box khuôn mặt

| Màu | Ý nghĩa |
|-----|---------|
| 🟢 Xanh lá | Học sinh đã quét thẻ QR hợp lệ |
| 🟡 Vàng cam | Học sinh biết nhưng CHƯA quét thẻ |
| 🔴 Đỏ | Người lạ / không trong danh sách |
| ⬜ Xám | Đang quét nhanh (Haar), chưa đối chiếu |

---

## 📊 Dashboard

```bash
python dashboard.py --open            # Báo cáo hôm nay
python dashboard.py --date 22/04/2026 # Ngày cụ thể
```

---

## ⚙️ Config quan trọng

| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `FACE_TOLERANCE` | Ngưỡng nhận diện | `0.5` |
| `HAAR_MIN_NEIGHBORS` | Độ nhạy Haar (thấp=nhạy hơn) | `4` |
| `FACE_SKIP_FRAMES` | face_recognition mỗi N frame | `5` |
| `CAPTURE_COOLDOWN` | Giây giữa 2 lần chụp vi phạm | `30` |
| `VIOLATION_COOLDOWN` | Giây giữa 2 lần ghi vi phạm | `60` |
| `SHOW_CLOCK` | Đồng hồ trên camera | `True` |
| `SHOW_STATS` | Thống kê trên camera | `True` |

*STEM Project — Hệ thống điểm danh QR Code + Nhận diện khuôn mặt v3.0*
