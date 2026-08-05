"""
=================================================
  HE THONG DIEM DANH QR CODE + NHAN DIEN MAT
  v3.0 — MERGE v1-modified + v2
=================================================
TU V1-MODIFIED (tinh nang ban them vao):
  [+] FaceEngine 2 lop: Haar nhanh moi frame + face_recognition dinh ky
  [+] Nhan dien NGUOI LA (stranger) - chup + ghi vi pham tu dong
  [+] Sidebar thong tin dep ben canh box khuon mat
  [+] Logic vi pham 3 muc: ok / warn (chua quet) / stranger

TU V2 (bug fix + tinh nang v2):
  [+] BUG FIX: vong lap vo han khi camera mat ket noi -> MAX_CAM_FAILS
  [+] BUG FIX: RAM leak recently_scan, last_results -> don dep dinh ky
  [+] BUG FIX: Unicode crash tren Windows console -> ASCII fallback
  [+] BUG FIX: mat du lieu khi Sheets loi -> SQLite offline buffer
  [+] Phim tat: S=thong ke, R=reload, P=pause mat, A=them mat live
  [+] Tong ket cuoi ngay tu dong len sheet TongKet
  [+] Kiem tra mang realtime hien tren camera
  [+] draw_banner dep voi fade-out
  [+] draw_clock, draw_stats tren camera
  [+] Auto sync offline -> Sheets background

Cach chay:
    python main.py
"""

import cv2
from pyzbar import pyzbar
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, date
import time
import sys
import os
import sqlite3
import threading
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import queue

from config import (
    SPREADSHEET_NAME, SHEET_LOG, SHEET_STUDENTS, SHEET_VIOLATIONS, SHEET_SUMMARY,
    COOLDOWN_SECONDS, CAMERA_INDEX, CREDENTIALS_FILE, SPREADSHEET_ID,
    FACE_GRACE_SECONDS, VIOLATION_COOLDOWN, FACE_SKIP_FRAMES,
    AUTO_SYNC_INTERVAL, BANNER_DURATION, DB_PATH,
    STREAM_ENABLE, STREAM_PORT, STREAM_QUALITY, STREAM_FPS
)
from utils import (
    load_students, find_student,
    beep_success, beep_error,
    draw_overlay, draw_banner, draw_clock, draw_stats,
    print_log, NetworkChecker
)
from face_engine import FaceEngine

try:
    import face_recognition as _FR_CHECK
    FACE_LIB_OK = True
except ImportError:
    FACE_LIB_OK = False


# =============================================================================
# MJPEG Stream Server — Phat giao dien da xu ly ve thiet bi camera
# =============================================================================
_stream_lock        = threading.Lock()
_stream_frame_bytes: bytes = b""
_stream_interval    = 1.0 / max(STREAM_FPS, 1)


def _push_stream_frame(frame_bgr):
    """Goi sau khi ve xong overlay — encode JPEG va luu vao buffer chia se."""
    global _stream_frame_bytes
    if not STREAM_ENABLE:
        return
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
    if not ok:
        return
    with _stream_lock:
        _stream_frame_bytes = buf.tobytes()


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler phuc vu trang HTML wrapper va luong MJPEG."""

    def log_message(self, fmt, *args):   # tat Apache-style log trong console
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/stream":
            self._serve_mjpeg()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        html = (
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Diem Danh STEM - Live</title>"
            "<style>"
            "body{margin:0;background:#0d1117;display:flex;flex-direction:column;"
            "align-items:center;justify-content:center;min-height:100vh;"
            "color:#c9d1d9;font-family:sans-serif;}"
            "img{width:100%;max-width:1280px;border:2px solid #238636;"
            "border-radius:8px;box-shadow:0 0 24px #0f0a;}"
            "p{margin:10px;font-size:12px;opacity:.6;}"
            "</style></head><body>"
            "<img src='/stream' />"
            "<p>He Thong Diem Danh STEM &mdash; Preview truc tiep (lam moi tu dong)</p>"
            "</body></html>"
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self):
        """Phat lien tuc frame JPEG theo chuan multipart/x-mixed-replace."""
        global _stream_frame_bytes
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last_push = 0.0
        try:
            while True:
                now = time.time()
                if now - last_push < _stream_interval:
                    time.sleep(0.005)
                    continue
                with _stream_lock:
                    data = _stream_frame_bytes
                if not data:
                    time.sleep(0.05)
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                )
                self.wfile.write(header + data + b"\r\n")
                self.wfile.flush()
                last_push = now
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def _start_stream_server():
    """Khoi dong MJPEG HTTP server trong daemon thread."""
    if not STREAM_ENABLE:
        return
    server = HTTPServer(("", STREAM_PORT), _MJPEGHandler)
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "<IP_may_tinh>"
    print(f"[STREAM] Preview server bat dau:")
    print(f"  May tinh nay : http://localhost:{STREAM_PORT}")
    print(f"  Thiet bi cam : http://{local_ip}:{STREAM_PORT}  (mo trinh duyet)")
    print()


# =============================================================================
# Ket noi Google Sheets
# =============================================================================
def connect_sheets():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    try:
        creds  = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        ss     = client.open_by_key(SPREADSHEET_ID) if SPREADSHEET_ID \
                 else client.open(SPREADSHEET_NAME)

        sheet_log = ss.worksheet(SHEET_LOG)
        sheet_stu = ss.worksheet(SHEET_STUDENTS)
        sheet_vio = ss.worksheet(SHEET_VIOLATIONS)

        # Tao sheet TongKet neu chua co
        try:
            sheet_sum = ss.worksheet(SHEET_SUMMARY)
        except gspread.WorksheetNotFound:
            sheet_sum = ss.add_worksheet(title=SHEET_SUMMARY, rows=500, cols=8)
            sheet_sum.append_row(
                ["Ngay", "Tong HS", "Co Mat", "Vang Mat",
                 "Vi Pham", "Nguoi La", "Gio Bat Dau", "Gio Ket Thuc"],
                value_input_option="RAW"
            )
            print(f"[OK] Da tao sheet '{SHEET_SUMMARY}'.")

        print(f"[OK] Da ket noi Google Sheets: '{ss.title}'")
        return sheet_log, sheet_stu, sheet_vio, sheet_sum

    except gspread.WorksheetNotFound as e:
        print(f"[!!] Khong tim thay sheet: {e}")
        print("     Chay 'python setup_sheets.py' truoc.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[!!] Khong tim thay file '{CREDENTIALS_FILE}'.")
        sys.exit(1)
    except Exception as e:
        print(f"[!!] Loi ket noi Sheets: {e}")
        sys.exit(1)


# =============================================================================
# SQLite offline buffer
# =============================================================================
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS offline_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thoi_gian TEXT, ma_hs TEXT, ho_ten TEXT, trang_thai TEXT,
            synced INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS offline_violation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thoi_gian TEXT, ma_hs TEXT, ho_ten TEXT, mo_ta TEXT,
            ten_file TEXT, duong_dan TEXT, synced INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def save_to_db(conn, rows, synced=0):
    conn.executemany(
        "INSERT INTO offline_log (thoi_gian,ma_hs,ho_ten,trang_thai,synced) VALUES(?,?,?,?,?)",
        [(r[0], r[1], r[2], r[3], synced) for r in rows]
    )
    conn.commit()


def save_violation_to_db(conn, row, synced=0):
    conn.execute(
        "INSERT INTO offline_violation (thoi_gian,ma_hs,ho_ten,mo_ta,ten_file,duong_dan,synced)"
        " VALUES(?,?,?,?,?,?,?)", row + [synced]
    )
    conn.commit()


def sync_offline_to_sheets(conn, sheet_log, sheet_vio):
    rows = conn.execute(
        "SELECT id,thoi_gian,ma_hs,ho_ten,trang_thai FROM offline_log"
        " WHERE synced=0 ORDER BY id LIMIT 50"
    ).fetchall()
    if rows:
        try:
            sheet_log.append_rows([[r[1],r[2],r[3],r[4]] for r in rows],
                                   value_input_option="RAW")
            ids = [r[0] for r in rows]
            conn.execute(
                f"UPDATE offline_log SET synced=1 WHERE id IN ({','.join('?'*len(ids))})", ids
            )
            conn.commit()
            print(f"[SYNC] Dong bo {len(rows)} ban ghi log.")
        except Exception as e:
            print(f"[SYNC] Loi log: {e}")

    vrows = conn.execute(
        "SELECT id,thoi_gian,ma_hs,ho_ten,mo_ta,ten_file,duong_dan"
        " FROM offline_violation WHERE synced=0 LIMIT 20"
    ).fetchall()
    if vrows:
        try:
            sheet_vio.append_rows([[r[1],r[2],r[3],r[4],r[5],r[6]] for r in vrows],
                                   value_input_option="RAW")
            vids = [r[0] for r in vrows]
            conn.execute(
                f"UPDATE offline_violation SET synced=1 WHERE id IN ({','.join('?'*len(vids))})", vids
            )
            conn.commit()
            print(f"[SYNC] Dong bo {len(vrows)} vi pham.")
        except Exception as e:
            print(f"[SYNC] Loi vi pham: {e}")


# =============================================================================
# Cache log hom nay
# =============================================================================
def load_today_log(sheet_log) -> dict:
    today = date.today().strftime("%d/%m/%Y")
    try:
        records = sheet_log.get_all_values()
    except Exception as e:
        print(f"[!!] Khong tai duoc log hom nay: {e}")
        return {}
    count_map = {}
    for row in records[1:]:
        if len(row) >= 2 and row[0].startswith(today):
            sid = row[1]
            count_map[sid] = count_map.get(sid, 0) + 1
    return count_map


def get_status_cache(count_map: dict, student_id: str) -> str:
    if not student_id:
        return "Vao"
    return "Ra" if count_map.get(str(student_id).strip(), 0) % 2 == 1 else "Vao"


# =============================================================================
# Ghi Sheets (fallback SQLite)
# =============================================================================
def flush_qr_batch(sheet_log, conn, batch: list) -> bool:
    if not batch:
        return True
    try:
        sheet_log.append_rows(batch, value_input_option="RAW")
        save_to_db(conn, batch, synced=1) # Log locally as already synced
        return True
    except Exception as e:
        print(f"[!!] Loi ghi QR Sheets (luu offline): {e}")
        save_to_db(conn, batch, synced=0) # Log locally as unsynced
        return False


def log_violation(sheet_vio, conn, student_id, student_name, photo_path, reason="Khong deo the"):
    ts       = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    filename = os.path.basename(photo_path)
    row = [ts, student_id, student_name, reason, filename, photo_path]
    try:
        sheet_vio.append_row(row, value_input_option="RAW")
        print(f"  [VI PHAM] {ts} | {student_id} | {student_name} | {reason}")
        save_violation_to_db(conn, row, synced=1) # Log locally as already synced
    except Exception as e:
        print(f"  [!!] Loi ghi vi pham (luu offline): {e}")
        save_violation_to_db(conn, row, synced=0) # Log locally as unsynced


def write_daily_summary(sheet_sum, count_map, total_students,
                         violation_count, stranger_count, start_time):
    today    = date.today().strftime("%d/%m/%Y")
    present  = sum(1 for c in count_map.values() if c % 2 == 1)
    absent   = total_students - present
    end_ts   = datetime.now().strftime("%H:%M:%S")
    start_ts = start_time.strftime("%H:%M:%S")
    # [v3] Them cot Nguoi La
    row = [today, total_students, present, absent,
           violation_count, stranger_count, start_ts, end_ts]
    try:
        sheet_sum.append_row(row, value_input_option="RAW")
        print(f"[OK] Da ghi tong ket ngay {today} len sheet '{SHEET_SUMMARY}'.")
    except Exception as e:
        print(f"[!!] Khong ghi duoc tong ket: {e}")


# =============================================================================
# Vong lap chinh
# =============================================================================
def run():
    print("=" * 60)
    print("  DIEM DANH STEM")
    print("  Haar Cascade + face_recognition + Offline Buffer")
    print("=" * 60)

    start_time = datetime.now()

    # Ket noi Sheets
    sheet_log, sheet_stu, sheet_vio, sheet_sum = connect_sheets()
    students = load_students(sheet_stu)
    print(f"[OK] Da tai {len(students)} hoc sinh.\n")

    conn      = get_db()
    count_map = load_today_log(sheet_log)
    print(f"[OK] Cache log: {sum(count_map.values())} ban ghi hom nay.\n")

    net_checker = NetworkChecker(check_interval=10)

    print("[FACE] Dang load Haar Cascade + khuon mat tu data/stuface/...")
    face_eng = FaceEngine(students)
    print()

    # Mo camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!!] Khong mo duoc camera (index={CAMERA_INDEX}).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    print("=" * 60)
    print("  CAMERA SAN SANG")
    print("  QR   : Quet the -> diem danh tu dong")
    print("  FACE : Haar bat mat moi frame")
    print("         face_recognition nhan dien dinh ky")
    print("         Phat hien nguoi la tu dong")
    print("  Phim : Q/ESC=Thoat  S=Thong ke  R=Reload")
    print("         P=Pause mat  A=Them mat live")
    print("=" * 60)

    # Tracking
    cooldown_map    = {}
    vio_cooldown    = {}
    recently_scan   = {}
    last_results    = {}
    fast_locations  = []    # [v1-mod] Haar locations moi frame
    face_results    = []    # face_recognition results dinh ky

    # [XAC MINH] pending_verify: luu thong tin quet the, CHO xac minh mat
    # Cau truc: {card_id: {"scan_time", "name", "ts", "status", "batch_row"}}
    # Chi ghi vao Sheets/count_map KHI khuon mat KHOP. Neu khong khop -> huy.
    VERIFY_WINDOW   = 15   # Giay cho doi nhan dien mat sau khi quet the
    pending_verify  = {}   # {card_id: dict}

    banner_msg   = ""
    banner_color = (0, 220, 80)
    banner_end   = 0.0
    face_paused  = False

    violation_count = 0
    stranger_count  = 0   # [v3] dem nguoi la
    last_sync_time  = time.time()

    frame_count    = 0
    cam_fail_count = 0
    MAX_CAM_FAILS  = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            cam_fail_count += 1
            if cam_fail_count >= MAX_CAM_FAILS:
                print(f"\n[!!] Camera mat ket noi sau {MAX_CAM_FAILS} lan. Thoat.")
                break
            time.sleep(0.05)
            continue
        cam_fail_count = 0

        now        = time.time()
        frame_count += 1

        # Don dep RAM dinh ky
        if frame_count % 300 == 0:
            cutoff       = now - 600
            recently_scan = {k: v for k, v in recently_scan.items() if v > cutoff}
            if len(last_results) > 10:
                for k in list(last_results.keys())[:-10]:
                    del last_results[k]

        # Background sync
        if net_checker.is_online and (now - last_sync_time) > AUTO_SYNC_INTERVAL:
            last_sync_time = now
            threading.Thread(
                target=sync_offline_to_sheets,
                args=(conn, sheet_log, sheet_vio), daemon=True
            ).start()

        # =====================================================================
        # BUOC 1: QUET QR CODE
        # =====================================================================
        decoded = pyzbar.decode(frame)
        batch   = []

        for obj in decoded:
            raw = obj.data.decode("utf-8", errors="replace").strip()

            if now - cooldown_map.get(raw, 0) < COOLDOWN_SECONDS:
                continue
            cooldown_map[raw] = now

            name = find_student(students, raw)
            if name is None:
                beep_error()
                last_results[raw] = {"text": raw, "status": "Khong tim thay",
                                     "name": "???", "ok": False}
                banner_msg   = f"The khong hop le: {raw[:20]}"
                banner_color = (0, 0, 200)
                banner_end   = now + BANNER_DURATION
                continue

            # Tinh trang (Vao/Ra) dua tren count_map HIEN TAI (chua cap nhat)
            status = get_status_cache(count_map, raw)
            ts     = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            # [XAC MINH] Chi luu vao hang doi, CHUA ghi Sheets / count_map
            pending_verify[raw] = {
                "scan_time": now,
                "name":      name,
                "ts":        ts,
                "status":    status,
                "batch_row": [ts, raw, name, status],
            }
            print(f"  [XAC MINH] Da quet the: {raw} ({name}) - Dang cho nhan dien mat...")
            beep_success()

            last_results[raw] = {"text": raw, "status": status,
                                  "name": name, "ok": True}

            status_vn    = "VAO" if status == "Vao" else "RA"
            banner_msg   = f"{status_vn}: {name} | Huong mat vao camera..."
            banner_color = (0, 200, 80) if status == "Vao" else (0, 160, 255)
            banner_end   = now + VERIFY_WINDOW

        # batch duoc flush SAU khi xac minh (xem phan XAC MINH ben duoi)

        # =====================================================================
        # BUOC 2A: HAAR CASCADE — phat hien mat nhanh moi frame [v1-mod]
        # =====================================================================
        fast_locations = face_eng.detect_locations(frame)

        # =====================================================================
        # BUOC 2B: FACE_RECOGNITION — nhan dien dinh ky [v1-mod + v2]
        # =====================================================================
        if (not face_paused and FACE_LIB_OK
                and frame_count % FACE_SKIP_FRAMES == 0):
            face_results = face_eng.recognize(frame)

            grace_ids = {
                sid for sid, t in recently_scan.items()
                if (now - t) < FACE_GRACE_SECONDS
            }

            # [XAC MINH] Don dep pending_verify HET HAN -> huy, khong ghi
            expired = {
                k: v for k, v in pending_verify.items()
                if (now - v["scan_time"]) >= VERIFY_WINDOW
            }
            for card_id, entry in expired.items():
                print(f"  [HUY] The '{card_id}' ({entry['name']}) het {VERIFY_WINDOW}s, khong nhan dien duoc mat -> HUY GHI NHAN")
                banner_msg   = f"[HUY] {entry['name']}: Khong xac minh duoc mat!"
                banner_color = (0, 100, 200)
                banner_end   = now + BANNER_DURATION
            pending_verify = {
                k: v for k, v in pending_verify.items()
                if (now - v["scan_time"]) < VERIFY_WINDOW
            }

            for fr in face_results:
                sid    = fr["student_id"]
                name   = fr["name"]
                status = fr["status"]

                # ── [XAC MINH] So sanh the vua quet vs khuon mat nhan dien ──
                if sid and pending_verify:
                    # Tim the con trong cua so xac minh khop voi mat nay
                    matched_card = None
                    for card_id, entry in list(pending_verify.items()):
                        if card_id.strip().upper() == sid.strip().upper():
                            matched_card = card_id
                            break

                    if matched_card:
                        # ==== KHOP ==== -> commit ghi nhan diem danh ====
                        entry = pending_verify.pop(matched_card)
                        # Ghi vao count_map (cap nhat trang thai)
                        count_map[matched_card] = count_map.get(matched_card, 0) + 1
                        # Ghi vao Sheets / SQLite
                        flush_qr_batch(sheet_log, conn, [entry["batch_row"]])
                        # Cap nhat recently_scan de bo qua canh bao chua quet
                        recently_scan[matched_card] = now
                        print_log(entry["ts"], matched_card, entry["name"], entry["status"])
                        print(f"  [KHOP] The '{matched_card}' - Mat '{sid}' ({name}) -> XAC NHAN DIEM DANH")
                        banner_msg   = f"[OK] KHOP: {name} ({matched_card})"
                        banner_color = (0, 220, 80)
                        banner_end   = now + BANNER_DURATION * 1.5

                    else:
                        # ==== KHONG KHOP ==== -> huy ca 2, ghi vi pham ====
                        for card_id, entry in list(pending_verify.items()):
                            card_name = entry["name"]
                            key_mismatch = f"MISMATCH_{card_id}"
                            if now - vio_cooldown.get(key_mismatch, 0) < VIOLATION_COOLDOWN:
                                continue
                            vio_cooldown[key_mismatch] = now
                            violation_count += 1
                            # Xoa khoi hang doi -> KHONG ghi count_map, KHONG ghi Sheets
                            pending_verify.pop(card_id, None)
                            print(f"  [GIAN LAN] The '{card_id}' ({card_name}) - Mat '{sid}' ({name}) -> KHONG KHOP! HUY GHI NHAN CA 2")
                            path = face_eng.capture_violation(
                                frame.copy(), card_id, card_name,
                                reason=f"GIAN LAN: The={card_id} Mat={sid}"
                            )
                            if path:
                                log_violation(
                                    sheet_vio, conn, card_id, card_name, path,
                                    reason=f"Gian lan: the {card_id} nhung mat {sid} ({name})"
                                )
                            beep_error()
                            banner_msg   = f"[!!] KHONG KHOP: The={card_id} / Mat={sid} -> HUY"
                            banner_color = (0, 0, 220)
                            banner_end   = now + BANNER_DURATION * 2
                            break

                # ── Ket thuc XAC MINH ──

                # [v1-mod] Hoc sinh chua quet QR
                if status == "warn" and sid not in grace_ids:
                    if now - vio_cooldown.get(sid, 0) < VIOLATION_COOLDOWN:
                        continue
                    vio_cooldown[sid] = now
                    violation_count  += 1
                    path = face_eng.capture_violation(
                        frame.copy(), sid, name, reason="CHUA QUET THE"
                    )
                    if path:
                        log_violation(sheet_vio, conn, sid, name, path,
                                      reason="Chua quet the")
                        beep_error()
                        banner_msg   = f"CANH BAO: {name} chua quet the!"
                        banner_color = (0, 60, 230)
                        banner_end   = now + BANNER_DURATION * 1.5

                # [v1-mod] Nguoi la
                elif status == "stranger":
                    key_stranger = f"STRANGER_{int(now/VIOLATION_COOLDOWN)}"
                    if now - vio_cooldown.get(key_stranger, 0) < VIOLATION_COOLDOWN:
                        continue
                    vio_cooldown[key_stranger] = now
                    stranger_count += 1
                    path = face_eng.capture_violation(
                        frame.copy(), "STRANGER", "Nguoi la", reason="NGUOI LA"
                    )
                    if path:
                        log_violation(sheet_vio, conn, "STRANGER", "Nguoi la", path,
                                      reason="Nguoi la - khong trong danh sach")
                        beep_error()
                        banner_msg   = "CANH BAO: Phat hien nguoi la!"
                        banner_color = (0, 0, 200)
                        banner_end   = now + BANNER_DURATION * 1.5

        # =====================================================================
        # BUOC 3: VE OVERLAY
        # =====================================================================
        result_display = {"text": "", "status": "", "name": "", "ok": True}
        if last_results:
            result_display = list(last_results.values())[-1]

        frame = draw_overlay(frame, decoded, result_display)

        # [v1-mod] Ve overlay Haar + face_recognition
        grace_ids_draw = {
            sid for sid, t in recently_scan.items()
            if (now - t) < FACE_GRACE_SECONDS
        }
        frame = face_eng.draw_overlay(frame, fast_locations, face_results, grace_ids_draw)

        # [v2] HUD
        frame = draw_clock(frame)
        frame = draw_stats(frame, count_map, len(students))

        # [v2] Banner fade-out
        if now < banner_end and banner_msg:
            alpha = min(1.0, (banner_end - now) / 0.5)
            frame = draw_banner(frame, banner_msg, banner_color, alpha)

        # Trang thai
        h_f, w_f = frame.shape[:2]
        if face_paused:
            cv2.putText(frame, "[P] Nhan dien mat: TAM DUNG",
                        (10, h_f - 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

        net_label = "[ONLINE]" if net_checker.is_online else "[OFFLINE - Luu cuc bo]"
        net_color = (0, 200, 80) if net_checker.is_online else (0, 100, 255)
        cv2.putText(frame, net_label,
                    (w_f - 260, h_f - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, net_color, 1)

        # HUD Haar + recog
        n_haar = len(fast_locations)
        n_recg = len(face_results)
        n_qr   = len(decoded)
        cv2.putText(frame,
                    f"Haar:{n_haar}  Nhan:{n_recg}  QR:{n_qr}  Nguoi la:{stranger_count}",
                    (10, h_f - 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

        cv2.imshow("He Thong Diem Danh  v3.0", frame)

        # =====================================================================
        # PHIM TAT
        # =====================================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord("s"):
            present = sum(1 for c in count_map.values() if c % 2 == 1)
            print("\n" + "=" * 44)
            print(f"  THONG KE  {date.today().strftime('%d/%m/%Y')}")
            print(f"  Tong hoc sinh : {len(students)}")
            print(f"  Co mat        : {present}")
            print(f"  Vang mat      : {len(students) - present}")
            print(f"  Vi pham       : {violation_count}")
            print(f"  Nguoi la      : {stranger_count}")
            print("=" * 44 + "\n")

        elif key == ord("r"):
            print("\n[RELOAD] Dang tai lai...")
            try:
                students  = load_students(sheet_stu)
                count_map = load_today_log(sheet_log)
                face_eng.students = students
                face_eng.reload()
                print(f"[RELOAD] Da tai {len(students)} hoc sinh.\n")
                banner_msg   = "Da tai lai danh sach!"
                banner_color = (0, 200, 80)
                banner_end   = now + BANNER_DURATION
            except Exception as e:
                print(f"[RELOAD] Loi: {e}\n")

        elif key == ord("p"):
            face_paused = not face_paused
            print(f"[FACE] {'TAM DUNG' if face_paused else 'TIEP TUC'} nhan dien mat.")

        elif key == ord("a"):
            sid_input = input("[ADD FACE] Nhap MaHS: ").strip().upper()
            if sid_input:
                if face_eng.add_face_live(frame.copy(), sid_input):
                    banner_msg   = f"Da them khuon mat: {sid_input}"
                    banner_color = (200, 180, 0)
                    banner_end   = now + BANNER_DURATION
                else:
                    banner_msg   = "Khong tim thay mat trong frame!"
                    banner_color = (0, 0, 200)
                    banner_end   = now + BANNER_DURATION

    # Tong ket cuoi ngay
    write_daily_summary(sheet_sum, count_map, len(students),
                         violation_count, stranger_count, start_time)

    # Dong bo cuoi
    if net_checker.is_online:
        print("[SYNC] Dong bo lan cuoi...")
        sync_offline_to_sheets(conn, sheet_log, sheet_vio)

    conn.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] He thong da dong. Tam biet!")


# =============================================================================
if __name__ == "__main__":
    run()
