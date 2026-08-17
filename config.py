import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Database cuc bo (SQLite) --------------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")

# ── Google Sheets -------------------------------------------------------------
SPREADSHEET_NAME = "Diem Danh STEM"
SPREADSHEET_ID   = os.environ.get("STEM_SPREADSHEET_ID",
                                   "1CLE7GcJNAiJe1irURuCWzq9WaSPZJPLaoufW9lXythU")

SHEET_LOG        = "Log"
SHEET_STUDENTS   = "HocSinh"
SHEET_VIOLATIONS = "PhamLoi"
SHEET_SUMMARY    = "TongKet"        # [v2] Tong ket cuoi ngay

# [fix-security] KHONG hard-code ten file credentials trong code nua —
# file JSON cu (chua khoa Google that) tung bi commit len GitHub public va
# da phai thu hoi. Tu nay, duong dan file credentials PHAI duoc truyen qua
# bien moi truong STEM_CREDENTIALS_FILE, tro toi 1 file nam NGOAI thu muc
# du an (VD: C:\Users\<ban>\secrets\credentials.json) de khong bao gio bi
# commit nham len Git.
#
# Cach dat bien moi truong tren Windows (CMD), chay 1 lan roi dong lai mo lai:
#     setx STEM_CREDENTIALS_FILE "C:\Users\ADMIN\secrets\credentials.json"
# Hoac dat tam thoi cho phien lam viec hien tai:
#     set STEM_CREDENTIALS_FILE=C:\Users\ADMIN\secrets\credentials.json
CREDENTIALS_FILE = os.environ.get("STEM_CREDENTIALS_FILE", "")

if not CREDENTIALS_FILE:
    print("[CANH BAO] Chua dat bien moi truong STEM_CREDENTIALS_FILE.")
    print("            Xem huong dan trong config.py de biet cach dat.")
elif not os.path.isfile(CREDENTIALS_FILE):
    print(f"[CANH BAO] Khong tim thay file credentials tai: {CREDENTIALS_FILE}")
else:
    try:
        _inside_project = os.path.commonpath(
            [os.path.abspath(CREDENTIALS_FILE), BASE_DIR]
        ) == BASE_DIR
    except ValueError:
        _inside_project = False  # O khac o dia -> chac chan khong nam trong du an
    if _inside_project:
        print("[CANH BAO] File credentials dang nam BEN TRONG thu muc du an.")
        print("            Neu du an nay se push len GitHub, hay chuyen file ra")
        print("            ngoai thu muc du an de tranh lo khoa nhu truoc day.")

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

# ── [v4] Kiem tra trang phuc (uniform_checker.py) -----------------------------
UNIFORM_CHECK_ENABLE   = True   # Bat/tat kiem tra trang phuc tich hop
UNIFORM_CHECK_INTERVAL = 15     # Chi chay HOG+AI moi N frame (giam tai CPU)
UNIFORM_VIOLATION_COOLDOWN = 90 # Giay — tranh chup lap lai lien tuc cung 1 nguoi

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
