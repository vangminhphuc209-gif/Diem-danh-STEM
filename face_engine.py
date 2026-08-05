"""
=================================================
  FACE_ENGINE.PY - NHAN DIEN KHUON MAT
  v3.0 — MERGE v1-modified + v2
=================================================
MERGE TU V1-MODIFIED (tinh nang moi cua ban):
  [+] Hai lop nhan dien:
        Lop 1 (moi frame) : Haar Cascade -> bat tat ca mat, ve box xam nhanh
        Lop 2 (dinh ky)   : face_recognition -> doi chieu danh tinh
  [+] Nhan dien NGUOI LA (stranger) - chup va ghi vi pham
  [+] Status 3 muc: ok / warn / stranger
  [+] Sidebar thong tin dep ben canh box
  [+] 4 goc khung nhan dep thay vi rectangle phu

GIU TU V2 (bug fix + tinh nang v2):
  [+] Thread-safe Lock khi add_face_live / reload
  [+] add_face_live(): them mat moi trong runtime (phim A)
  [+] reload(): hot-reload khong can thoat chuong trinh
  [+] Confidence bar duoi khung mat
  [+] Clamp toa do tranh ve ra ngoai frame (BUG FIX)
  [+] Kiem tra frame hop le truoc khi xu ly (BUG FIX)
  [+] CAPTURE_COOLDOWN tranh spam anh vi pham
  [+] save_violation_photo: khung do + watermark dep hon

Mau sac box:
  Xanh la  : Hoc sinh DA QUET the QR
  Vang cam : Hoc sinh biet nhung CHUA QUET the
  Do        : Nguoi la / khong co trong danh sach
  Xam       : Dang quet nhanh (Haar), chua doi chieu danh tinh
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
import threading

try:
    import face_recognition
    FACE_LIB = "face_recognition"
except ImportError:
    FACE_LIB = None
    print("[WARN] face_recognition chua duoc cai.")
    print("       pip install face_recognition")
    print("       He thong van chay voi Haar Cascade (khong nhan dien duoc danh tinh).\n")

from config import (
    FACE_DB_PATH, VIOLATION_PATH, FACE_TOLERANCE, FACE_SCALE,
    HAAR_SCALE_FACTOR, HAAR_MIN_NEIGHBORS, HAAR_MIN_SIZE,
    CAPTURE_COOLDOWN,
    COLOR_FACE_OK, COLOR_FACE_WARN, COLOR_FACE_STRANGER, COLOR_FACE_SCAN
)


# =============================================================================
class FaceEngine:
    """
    Nhan dien khuon mat 2 lop:
      Lop 1 (moi frame) : Haar Cascade  -> bat tat ca khuon mat, ve box nhanh
      Lop 2 (dinh ky)   : face_recognition -> doi chieu danh tinh & stranger
    """

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, students: dict):
        self.students  = students     # {ma_hs: ho_ten}
        self.known_ids  = []
        self.known_encs = []
        self.available  = True        # Haar Cascade luon co san
        self._lock      = threading.Lock()  # [v2] thread-safe

        # Haar Cascade — phat hien mat moi frame
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("[WARN] Khong tai duoc Haar Cascade. Kiem tra cai dat OpenCV.")
            self.available = False

        # Cooldown chup anh {key: timestamp}
        self._capture_cooldown: dict = {}

        if FACE_LIB:
            self._load_known_faces()
        else:
            print("[FACE] Bo qua doi chieu danh tinh: face_recognition chua cai.")

    # -------------------------------------------------------------------------
    def _load_known_faces(self):
        """Load va encode tat ca anh tu thu muc stuface/."""
        if not os.path.isdir(FACE_DB_PATH):
            os.makedirs(FACE_DB_PATH, exist_ok=True)
            print(f"[FACE] Da tao thu muc stuface: {FACE_DB_PATH}")
            print("       Dat anh hoc sinh vao day (ten file = MaHS.jpg)")
            return

        count  = 0
        failed = []
        for fname in sorted(os.listdir(FACE_DB_PATH)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in self.SUPPORTED_EXT:
                continue
            ma_hs = stem.strip().upper()
            path  = os.path.join(FACE_DB_PATH, fname)
            try:
                img  = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if not encs:
                    failed.append(fname)
                    continue
                with self._lock:
                    self.known_ids.append(ma_hs)
                    self.known_encs.append(encs[0])
                count += 1
            except Exception as e:
                print(f"[FACE] Loi load '{fname}': {e}")

        print(f"[FACE] Da load {count} khuon mat tu {FACE_DB_PATH}")
        if failed:
            print(f"[FACE] Khong tim thay mat trong: {', '.join(failed)}")

    # =========================================================================
    # LOP 1: HAAR CASCADE — phat hien nhanh moi frame
    # =========================================================================
    def detect_locations(self, frame_bgr) -> list:
        """
        [v1-mod] Quet nhanh toan bo khuon mat bang Haar Cascade.
        Tra ve list (top, right, bottom, left).
        """
        if not self.available or frame_bgr is None or frame_bgr.size == 0:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # tang tuong phan
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        return [(y, x + w, y + h, x) for (x, y, w, h) in faces]

    # =========================================================================
    # LOP 2: FACE_RECOGNITION — doi chieu danh tinh (theo ky)
    # =========================================================================
    def recognize(self, frame_bgr) -> list:
        """
        [v1-mod + v2] Nhan dien danh tinh + phan loai stranger.
        Tra ve list dict:
          {student_id, name, location, confidence, status}
          status: 'ok' (tam thoi, se cap nhat tu scanned_ids) |
                  'warn' (biet nhung chua quet QR) |
                  'stranger' (nguoi la)
        """
        if not FACE_LIB:
            return []

        # [v2 BUG FIX] Kiem tra frame hop le
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        with self._lock:
            if not self.known_encs:
                # Khong co ai trong DB -> tat ca deu la stranger
                ids_snap  = []
                encs_snap = []
            else:
                ids_snap  = list(self.known_ids)
                encs_snap = list(self.known_encs)

        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)

        try:
            locations = face_recognition.face_locations(small, model="hog")
            if not locations:
                return []
            encodings = face_recognition.face_encodings(small, locations)
        except Exception as e:
            print(f"[FACE] Loi nhan dien: {e}")
            return []

        results = []
        for enc, loc in zip(encodings, locations):
            top, right, bottom, left = [int(v / FACE_SCALE) for v in loc]

            result = {
                "student_id": None,
                "name":       "Nguoi la",
                "location":   (top, right, bottom, left),
                "confidence": 0.0,
                "status":     "stranger",   # [v1-mod] mac dinh la stranger
            }

            if encs_snap:
                distances  = face_recognition.face_distance(encs_snap, enc)
                best_idx   = int(np.argmin(distances))
                best_dist  = distances[best_idx]
                confidence = max(0.0, 1.0 - best_dist)

                if best_dist <= FACE_TOLERANCE:
                    sid  = ids_snap[best_idx]
                    name = self.students.get(sid, sid)
                    result.update({
                        "student_id": sid,
                        "name":       name,
                        "confidence": round(confidence, 2),
                        "status":     "warn",  # biet nhung chua quet QR
                    })

            results.append(result)
        return results

    # =========================================================================
    # VE OVERLAY — 2 lop hop nhat
    # =========================================================================
    def draw_overlay(self, frame, fast_locations: list,
                     recog_results: list, scanned_ids: set):
        """
        [v1-mod + v2] Ve toan bo overlay.
        - fast_locations : Haar boxes (ve xam neu chua co recog match)
        - recog_results  : ket qua face_recognition
        - scanned_ids    : tap hop ma_hs da quet QR hop le
        """
        h_frame, w_frame = frame.shape[:2]

        # Cap nhat status 'ok' cho hoc sinh da quet the
        updated = []
        for r in recog_results:
            r = dict(r)
            if r["student_id"] and r["student_id"] in scanned_ids:
                r["status"] = "ok"
            updated.append(r)

        # Ve Haar box xam cho mat chua match recog
        recog_locs = [r["location"] for r in updated]
        for (top, right, bottom, left) in fast_locations:
            # Clamp
            top    = max(0, top);    left   = max(0, left)
            bottom = min(h_frame, bottom); right = min(w_frame, right)
            matched = any(
                abs(top - rt) < 40 and abs(left - rl) < 40
                for (rt, rr, rb, rl) in recog_locs
            )
            if not matched:
                self._draw_box(frame, top, right, bottom, left,
                               COLOR_FACE_SCAN, "Dang quet...", None, None, None)

        # Ve box chinh thuc
        for r in updated:
            top, right, bottom, left = r["location"]
            # [v2 BUG FIX] Clamp toa do
            top    = max(0, top);    left   = max(0, left)
            bottom = min(h_frame, bottom); right = min(w_frame, right)

            sid    = r["student_id"]
            name   = r["name"]
            conf   = r["confidence"]
            status = r["status"]

            if status == "ok":
                color   = COLOR_FACE_OK
                vio_tag = "Co the"
            elif status == "warn":
                color   = COLOR_FACE_WARN
                vio_tag = "CHUA QUET THE!"
            else:
                color   = COLOR_FACE_STRANGER
                vio_tag = "NGUOI LA"
                sid     = "N/A"
                name    = "Nguoi la"

            self._draw_box(frame, top, right, bottom, left,
                           color, name, sid, vio_tag, conf)

            # [v2] Confidence bar nho duoi khung
            if conf and conf > 0 and status != "stranger":
                bar_w  = right - left
                filled = int(bar_w * conf)
                cv2.rectangle(frame,
                              (left, bottom + 2), (right, bottom + 6), (50,50,50), -1)
                cv2.rectangle(frame,
                              (left, bottom + 2), (left + filled, bottom + 6), color, -1)

        return frame

    # -------------------------------------------------------------------------
    def _draw_box(self, frame, top, right, bottom, left,
                  color, label, sid, vio_tag, conf):
        """
        [v1-mod] Ve khung 4-goc dep + sidebar thong tin.
        """
        h_frame, w_frame = frame.shape[:2]
        L = 18  # chieu dai canh goc

        # 4 goc
        corners_arms = [
            ((left,  top),    [(left+L, top),    (left,  top+L)]),
            ((right, top),    [(right-L, top),   (right, top+L)]),
            ((left,  bottom), [(left+L, bottom), (left,  bottom-L)]),
            ((right, bottom), [(right-L, bottom),(right, bottom-L)]),
        ]
        for corner, arms in corners_arms:
            for arm in arms:
                cv2.line(frame, corner, arm, color, 3)
        # Vien mong ben trong
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

        # Sidebar thong tin ben phai box
        sidebar_x = min(right + 8, w_frame - 155)
        sidebar_y = top

        info_lines = []
        if sid is not None:
            info_lines.append(("ID:", sid if sid != "N/A" else "---"))
            info_lines.append(("Ten:", label[:18]))
            if vio_tag:
                info_lines.append(("TT:", vio_tag))
            if conf:
                info_lines.append(("Conf:", f"{int(conf*100)}%"))
        else:
            info_lines.append(("TT:", label))

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.42
        fthick = 1
        line_h = 18
        pad    = 4

        max_w = 0
        for k, v in info_lines:
            tw, _ = cv2.getTextSize(f"{k} {v}", font, fscale, fthick)[0]
            max_w = max(max_w, tw)

        sb_x1 = sidebar_x
        sb_y1 = sidebar_y
        sb_x2 = min(sb_x1 + max_w + pad * 2, w_frame - 2)
        sb_y2 = sidebar_y + len(info_lines) * line_h + pad * 2

        # Nen mo
        ov = frame.copy()
        cv2.rectangle(ov, (sb_x1, sb_y1), (sb_x2, sb_y2), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (sb_x1, sb_y1), (sb_x2, sb_y2), color, 1)

        for i, (key, val) in enumerate(info_lines):
            y_pos = sidebar_y + pad + (i + 1) * line_h
            if y_pos > h_frame - 5:
                break
            kw = cv2.getTextSize(key + " ", font, fscale, fthick)[0][0]
            cv2.putText(frame, key, (sb_x1 + pad, y_pos),
                        font, fscale, (170, 170, 170), fthick)
            cv2.putText(frame, val, (sb_x1 + pad + kw, y_pos),
                        font, fscale, (255, 255, 255), fthick)

        # Nhan nho duoi box
        lbl_y = bottom + 16 if bottom + 20 < h_frame else top - 6
        lbl_y = max(12, min(lbl_y, h_frame - 5))
        cv2.putText(frame, label, (left, lbl_y), font, 0.5, color, 1)

    # =========================================================================
    # ADD FACE LIVE — [v2]
    # =========================================================================
    def add_face_live(self, frame_bgr, student_id: str) -> bool:
        """
        [v2] Them khuon mat moi tu frame camera ngay trong runtime.
        Luu anh vao stuface/ va cap nhat encoding trong RAM (thread-safe).
        """
        if not FACE_LIB:
            print("[FACE] Can face_recognition de them khuon mat.")
            return False
        ma_hs = student_id.strip().upper()
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        encs  = face_recognition.face_encodings(rgb)
        if not encs:
            print(f"[FACE] Khong tim thay mat trong frame khi add {ma_hs}")
            return False
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        save_path = os.path.join(FACE_DB_PATH, f"{ma_hs}.jpg")
        cv2.imwrite(save_path, frame_bgr)
        with self._lock:
            if ma_hs in self.known_ids:
                idx = self.known_ids.index(ma_hs)
                self.known_encs[idx] = encs[0]
            else:
                self.known_ids.append(ma_hs)
                self.known_encs.append(encs[0])
        print(f"[FACE] Da them khuon mat: {ma_hs} -> {save_path}")
        return True

    # =========================================================================
    # RELOAD — [v2]
    # =========================================================================
    def reload(self):
        """[v2] Hot-reload toan bo khuon mat tu stuface/."""
        with self._lock:
            self.known_ids  = []
            self.known_encs = []
        if FACE_LIB:
            self._load_known_faces()
        print("[FACE] Reload hoan tat.")

    # =========================================================================
    # CHUP ANH VI PHAM
    # =========================================================================
    def capture_violation(self, frame, student_id: str, student_name: str,
                          reason: str = "VI PHAM") -> Optional[str]:
        """
        [v1-mod + v2] Chup anh vi pham voi cooldown tranh spam.
        Tra ve filepath hoac None neu con trong cooldown.
        """
        key = (student_id or "stranger").replace("/", "_")
        now = datetime.now().timestamp()
        if now - self._capture_cooldown.get(key, 0) < CAPTURE_COOLDOWN:
            return None
        self._capture_cooldown[key] = now
        return FaceEngine.save_violation_photo(frame, student_id, student_name, reason)

    # -------------------------------------------------------------------------
    @staticmethod
    def save_violation_photo(frame, student_id: str, student_name: str,
                             reason: str = "VI PHAM") -> str:
        """
        [v1-mod + v2] Luu anh vi pham vao data/violations/.
        Watermark dep, khung do, JPEG quality 92.
        """
        os.makedirs(VIOLATION_PATH, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id  = (student_id or "stranger").replace("/", "_")
        filename = f"{ts}_{safe_id}.jpg"
        filepath = os.path.join(VIOLATION_PATH, filename)

        snap = frame.copy()
        h, w = snap.shape[:2]

        # Banner do phia duoi
        banner_h = 55
        ov = snap.copy()
        cv2.rectangle(ov, (0, h - banner_h), (w, h), (0, 0, 150), -1)
        cv2.addWeighted(ov, 0.70, snap, 0.30, 0, snap)

        ts_disp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(snap,
                    f"! {reason}: {student_name}  |  ID: {student_id or 'N/A'}",
                    (10, h - banner_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(snap, ts_disp,
                    (10, h - banner_h + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Khung do ngoai cung
        cv2.rectangle(snap, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)

        cv2.imwrite(filepath, snap, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  [CAM] Da luu anh vi pham: {filename}")
        return filepath


# =============================================================================
# Test truc tiep
# =============================================================================
if __name__ == "__main__":
    print("=" * 54)
    print("  FACE ENGINE v3 — TEST WEBCAM LIVE")
    print("  Haar Cascade + face_recognition")
    print("=" * 54)

    dummy_students = {}
    if os.path.isdir(FACE_DB_PATH):
        for f in os.listdir(FACE_DB_PATH):
            stem = os.path.splitext(f)[0].upper()
            dummy_students[stem] = stem

    engine       = FaceEngine(dummy_students)
    cap          = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count   = 0
    recog_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        fast_locs = engine.detect_locations(frame)
        if frame_count % 5 == 0 and FACE_LIB:
            recog_results = engine.recognize(frame)

        frame = engine.draw_overlay(frame, fast_locs, recog_results, set())
        cv2.putText(frame,
                    f"Haar: {len(fast_locs)}  |  Nhan dien: {len(recog_results)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("Face Engine v3 - Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
