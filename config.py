"""
=================================================
  CONFIG.PY - CAU HINH HE THONG DIEM DANH
  v3.0 — MERGE v1-modified + v2
=================================================
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Database cuc bo (SQLite) --------------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")

# ── Google Sheets -------------------------------------------------------------
SPREADSHEET_NAME = "Diem Danh STEM"
SPREADSHEET_ID   = "1CLE7GcJNAiJe1irURuCWzq9WaSPZJPLaoufW9lXythU"

SHEET_LOG        = "Log"
SHEET_STUDENTS   = "HocSinh"
SHEET_VIOLATIONS = "PhamLoi"
SHEET_SUMMARY    = "TongKet"        # [v2] Tong ket cuoi ngay

CREDENTIALS_FILE = "reflecting-site-494013-b3-9978cb2582bd.json"

# ── Camera -------------------------------------------------------------------
CAMERA_INDEX = 0   # 0 = webcam mac dinh, 1 = ngoai, hoac URL IP cam http://192.168.1.38:4747/video

# ── Logic diem danh ----------------------------------------------------------
COOLDOWN_SECONDS = 5

# [v2] Offline buffer: luu SQLite khi Sheets loi, dong bo khi co mang
OFFLINE_MODE       = False
AUTO_SYNC_INTERVAL = 60   # giay

# ── Nhan dien khuon mat ------------------------------------------------------
FACE_DB_PATH   = os.path.join(BASE_DIR, "data", "stuface")
VIOLATION_PATH = os.path.join(BASE_DIR, "data", "violations")

FACE_GRACE_SECONDS = 30     # Grace window sau khi quet QR (giay)
FACE_TOLERANCE     = 0.5    # Nguong face_recognition (0.4=chat, 0.6=thoang)
VIOLATION_COOLDOWN = 60     # Cooldown ghi vi pham (giay)
CAPTURE_COOLDOWN   = 30     # Cooldown chup anh (giay)

# [v2] Nhan dien face_recognition moi N frame
FACE_SKIP_FRAMES = 5
FACE_SCALE       = 0.5      # Ti le thu nho anh truoc khi nhan dien

# [v1-mod] Haar Cascade — phat hien mat nhanh moi frame
HAAR_SCALE_FACTOR  = 1.05
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (40, 40)   # pixel

# ── Mau sac (BGR) ------------------------------------------------------------
COLOR_SUCCESS       = (0, 220, 80)
COLOR_ERROR         = (0, 0, 220)
COLOR_INFO          = (220, 180, 0)
COLOR_VIOLATION     = (0, 80, 255)
COLOR_FACE_OK       = (50, 220, 50)     # Xanh la  - co the + da quet QR
COLOR_FACE_WARN     = (0, 200, 255)     # Vang cam - biet nhung chua quet
COLOR_FACE_STRANGER = (30, 30, 220)     # Do       - nguoi la
COLOR_FACE_SCAN     = (180, 180, 180)   # Xam      - dang quet Haar nhanh

# [v2] HUD tren camera
SHOW_CLOCK      = True
SHOW_STATS      = True
BANNER_DURATION = 3.0

# ── MJPEG Preview Stream Server ----------------------------------------------
# Bat/tat tinh nang stream giao dien da xu ly ve thiet bi camera
STREAM_ENABLE   = True
STREAM_PORT     = 8080      # Cong HTTP; thiet bi camera mo: http://<IP_may_tinh>:8080
STREAM_QUALITY  = 70        # Chat luong JPEG (0-100); 70 la can bang tot
STREAM_FPS      = 15        # FPS stream toi da
