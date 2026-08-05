# -*- coding: utf-8 -*-
"""
=============================================================
  MODULE PHAN TICH MAU SAC TRANG PHUC
  uniform_color_analyzer.py

  Phan tich mau sac ao va quan bang khong gian mau HSV.
  KHONG can training AI - chay tuc thi, on dinh cao.

  Quy dinh:
    Thu 2  : Trang phuc dan toc   (xu ly boi TFLite AI)
    Thu 3  : Ao TRANG + Quan TOI MAU
    Thu 4  : Tu do co co          (khong check tu dong)
    Thu 5  : Ao TRANG + Quan TOI MAU
    Thu 6  : Ao XANH DUONG (doan) + Quan TOI MAU
    Thu 7  : Nghi
    CN     : Nghi
=============================================================
"""

import cv2
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional


# ==================== CAU HINH MAU HSV ====================
# HSV: H(0-179), S(0-255), V(0-255) trong OpenCV

# Ao trang: Sac bao hoa thap (it mau), Do sang cao
WHITE_S_MAX = 60    # Saturation <= 60: gan nhu trang/xam sang
WHITE_V_MIN = 160   # Value >= 160: du sang (tranh mau xam toi)

# Ao xanh duong (doan vien): Hue trong vung xanh duong
BLUE_H_MIN  = 95    # Hue bat dau vung xanh duong
BLUE_H_MAX  = 135   # Hue ket thuc vung xanh duong
BLUE_S_MIN  = 60    # Du bao hoa (khong phai xam)
BLUE_V_MIN  = 40    # Khong qua toi

# Quan toi mau: Do sang (Value) thap
DARK_V_MAX  = 100   # Value <= 100: duoc coi la toi mau
DARK_RATIO_MIN = 0.45  # It nhat 45% diem anh trong vung quan phai la toi mau

# Nguong do chinh xac toi thieu de chap nhan ket qua
MIN_PIXEL_RATIO = 0.30  # 30% pixels phai khop moi duoc tin tuong


# ==================== KET QUA ====================
@dataclass
class ColorCheckResult:
    status: str          # "OK" / "FAIL" / "SKIP" / "UNCLEAR"
    is_shirt_ok: bool
    is_pants_ok: bool
    shirt_color: str     # Ten mau phat hien duoc
    pants_color: str     # "toi" / "sang" / "khong ro"
    shirt_ratio: float   # Ti le pixels khop voi mau quy dinh (0.0-1.0)
    pants_dark_ratio: float  # Ti le pixels toi mau o vung quan
    message: str
    debug_colors: dict   # Thong tin debug


# ==================== CLASS CHINH ====================
class UniformColorAnalyzer:
    """
    Phan tich trang phuc bang mau sac HSV.
    Ket hop voi uniform_checker.py de kiem tra day du.
    """

    # Chia khung hinh thanh vung ao va quan
    # Vung ao: tu 15% den 55% chieu cao (tranh mat, co)
    # Vung quan: tu 55% den 90% chieu cao (tranh giay)
    SHIRT_ZONE = (0.15, 0.55)
    PANTS_ZONE = (0.55, 0.90)

    # ----------------------------------------------------------------
    # HAM CHINH: KIEM TRA TRANG PHUC THEO NGAY
    # ----------------------------------------------------------------
    def check(self, frame: np.ndarray,
              body_box: Optional[Tuple] = None,
              weekday: Optional[int] = None) -> ColorCheckResult:
        """
        Kiem tra trang phuc dua tren mau sac.

        Args:
            frame    : Anh BGR tu camera
            body_box : (x, y, w, h) vung than nguoi. None = dung toan bo anh
            weekday  : 0=T2, 1=T3, 2=T4, 3=T5, 4=T6, 5=T7, 6=CN
                       None = tu dong lay ngay hom nay

        Returns:
            ColorCheckResult
        """
        if weekday is None:
            weekday = datetime.now().weekday()

        # Cat vung than nguoi
        body_region = self._crop_body(frame, body_box)

        # Lay vung ao va quan
        shirt_region = self._get_zone(body_region, *self.SHIRT_ZONE)
        pants_region = self._get_zone(body_region, *self.PANTS_ZONE)

        # Phan tich mau
        shirt_analysis = self._analyze_shirt_region(shirt_region)
        pants_analysis = self._analyze_pants_darkness(pants_region)

        # Ap dung quy dinh theo ngay
        return self._apply_rule(weekday, shirt_analysis, pants_analysis)

    # ----------------------------------------------------------------
    # CAC HAM PHAN TICH MAU
    # ----------------------------------------------------------------
    def _analyze_shirt_region(self, region: np.ndarray) -> dict:
        """Phan tich mau ao: trang, xanh duong, hay khac."""
        if region is None or region.size == 0:
            return {"detected": "unknown", "white_ratio": 0.0,
                    "blue_ratio": 0.0, "dominant_hsv": (0, 0, 0)}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # --- Kiem tra MAU TRANG ---
        # Trang: Saturation thap + Value cao
        white_mask = cv2.inRange(
            hsv,
            np.array([0,         0,         WHITE_V_MIN]),
            np.array([179,       WHITE_S_MAX, 255])
        )
        white_ratio = float(np.sum(white_mask > 0)) / total_pixels

        # --- Kiem tra MAU XANH DUONG ---
        blue_mask = cv2.inRange(
            hsv,
            np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
            np.array([BLUE_H_MAX, 255,        255])
        )
        blue_ratio = float(np.sum(blue_mask > 0)) / total_pixels

        # Mau trung binh de debug
        mean_hsv = cv2.mean(hsv)[:3]

        # Quyet dinh mau chinh
        if white_ratio >= MIN_PIXEL_RATIO:
            detected = "trang"
        elif blue_ratio >= MIN_PIXEL_RATIO:
            detected = "xanh_duong"
        elif white_ratio >= 0.15:
            detected = "co_trang_1_phan"  # Co trang nhung khong du
        else:
            detected = "khac"

        return {
            "detected"    : detected,
            "white_ratio" : round(white_ratio, 3),
            "blue_ratio"  : round(blue_ratio, 3),
            "dominant_hsv": tuple(int(x) for x in mean_hsv),
        }

    def _analyze_pants_darkness(self, region: np.ndarray) -> dict:
        """Kiem tra quan co toi mau khong."""
        if region is None or region.size == 0:
            return {"is_dark": False, "dark_ratio": 0.0, "mean_value": 255}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # Toi mau: Value thap
        dark_mask = cv2.inRange(
            hsv,
            np.array([0,   0,   0]),
            np.array([179, 255, DARK_V_MAX])
        )
        dark_ratio  = float(np.sum(dark_mask > 0)) / total_pixels
        mean_value  = float(np.mean(hsv[:, :, 2]))  # Trung binh do sang

        is_dark     = dark_ratio >= DARK_RATIO_MIN

        # Mo ta
        if dark_ratio >= 0.65:
            color_desc = "den/xanh den"
        elif dark_ratio >= 0.45:
            color_desc = "toi mau"
        elif mean_value > 170:
            color_desc = "sang mau"
        else:
            color_desc = "trung binh"

        return {
            "is_dark"    : is_dark,
            "dark_ratio" : round(dark_ratio, 3),
            "mean_value" : round(mean_value, 1),
            "color_desc" : color_desc,
        }

    # ----------------------------------------------------------------
    # AP DUNG QUY DINH THEO NGAY TRONG TUAN
    # ----------------------------------------------------------------
    def _apply_rule(self, weekday: int,
                    shirt: dict, pants: dict) -> ColorCheckResult:

        shirt_color = shirt["detected"]
        pants_desc  = pants["color_desc"]
        is_pants_dark = pants["is_dark"]

        debug = {
            "white_ratio"     : shirt["white_ratio"],
            "blue_ratio"      : shirt["blue_ratio"],
            "dark_pants_ratio": pants["dark_ratio"],
            "mean_pants_value": pants["mean_value"],
            "shirt_hsv_mean"  : shirt["dominant_hsv"],
        }

        # ----- THU 3 (weekday=1) va THU 5 (weekday=3): AO TRANG + QUAN TOI -----
        if weekday in (1, 3):
            is_shirt_ok = shirt_color in ("trang",)
            is_pants_ok = is_pants_dark

            if is_shirt_ok and is_pants_ok:
                return ColorCheckResult(
                    status="OK", is_shirt_ok=True, is_pants_ok=True,
                    shirt_color="Trang", pants_color=pants_desc,
                    shirt_ratio=shirt["white_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message=f"OK: Ao trang + quan toi ({pants['dark_ratio']*100:.0f}% toi)",
                    debug_colors=debug
                )
            elif not is_shirt_ok and not is_pants_ok:
                return ColorCheckResult(
                    status="FAIL", is_shirt_ok=False, is_pants_ok=False,
                    shirt_color=shirt_color, pants_color=pants_desc,
                    shirt_ratio=shirt["white_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message="SAI: Can ao trang + quan toi mau",
                    debug_colors=debug
                )
            elif not is_shirt_ok:
                return ColorCheckResult(
                    status="FAIL", is_shirt_ok=False, is_pants_ok=True,
                    shirt_color=shirt_color, pants_color=pants_desc,
                    shirt_ratio=shirt["white_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message=f"SAI AO: Can ao trang (phat hien: {shirt_color})",
                    debug_colors=debug
                )
            else:  # quan sai
                return ColorCheckResult(
                    status="FAIL", is_shirt_ok=True, is_pants_ok=False,
                    shirt_color="Trang", pants_color=pants_desc,
                    shirt_ratio=shirt["white_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message=f"SAI QUAN: Quan phai toi mau (hien tai: {pants_desc})",
                    debug_colors=debug
                )

        # ----- THU 6 (weekday=4): AO XANH DUONG (DOAN) + QUAN TOI -----
        elif weekday == 4:
            is_shirt_ok = shirt_color in ("xanh_duong",)
            is_pants_ok = is_pants_dark

            if is_shirt_ok and is_pants_ok:
                return ColorCheckResult(
                    status="OK", is_shirt_ok=True, is_pants_ok=True,
                    shirt_color="Xanh duong", pants_color=pants_desc,
                    shirt_ratio=shirt["blue_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message=f"OK: Ao doan xanh + quan toi ({shirt['blue_ratio']*100:.0f}% xanh)",
                    debug_colors=debug
                )
            elif not is_shirt_ok and not is_pants_ok:
                return ColorCheckResult(
                    status="FAIL", is_shirt_ok=False, is_pants_ok=False,
                    shirt_color=shirt_color, pants_color=pants_desc,
                    shirt_ratio=shirt["blue_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message="SAI: Can ao doan xanh + quan toi mau",
                    debug_colors=debug
                )
            elif not is_shirt_ok:
                conf_pct = shirt["blue_ratio"] * 100
                return ColorCheckResult(
                    status="FAIL", is_shirt_ok=False, is_pants_ok=is_pants_dark,
                    shirt_color=shirt_color, pants_color=pants_desc,
                    shirt_ratio=shirt["blue_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message=f"SAI AO: Can ao doan xanh duong (chi co {conf_pct:.0f}% xanh)",
                    debug_colors=debug
                )
            else:
                return ColorCheckResult(
                    status="FAIL", is_shirt_ok=True, is_pants_ok=False,
                    shirt_color="Xanh duong", pants_color=pants_desc,
                    shirt_ratio=shirt["blue_ratio"],
                    pants_dark_ratio=pants["dark_ratio"],
                    message=f"SAI QUAN: Quan phai toi mau (hien tai: {pants_desc})",
                    debug_colors=debug
                )

        # ----- CAC NGAY KHAC: SKIP -----
        else:
            day_names = {0: "Thu 2 (dan toc - AI xu ly)",
                         2: "Thu 4 (tu do co co)",
                         5: "Thu 7 (nghi)", 6: "CN (nghi)"}
            msg = day_names.get(weekday, "Khong yeu cau")
            return ColorCheckResult(
                status="SKIP", is_shirt_ok=True, is_pants_ok=True,
                shirt_color="N/A", pants_color="N/A",
                shirt_ratio=0.0, pants_dark_ratio=0.0,
                message=msg, debug_colors=debug
            )

    # ----------------------------------------------------------------
    # HO TRO CAT VUNG
    # ----------------------------------------------------------------
    @staticmethod
    def _crop_body(frame: np.ndarray,
                   body_box: Optional[Tuple]) -> np.ndarray:
        if body_box is None:
            return frame
        x, y, w, h = body_box
        x = max(0, x); y = max(0, y)
        return frame[y:y+h, x:x+w]

    @staticmethod
    def _get_zone(region: np.ndarray,
                  y_start_ratio: float,
                  y_end_ratio: float) -> np.ndarray:
        if region is None or region.size == 0:
            return region
        h = region.shape[0]
        y0 = int(h * y_start_ratio)
        y1 = int(h * y_end_ratio)
        return region[y0:y1, :]

    # ----------------------------------------------------------------
    # VE KET QUA LEN FRAME CAMERA (De tich hop vao main.py)
    # ----------------------------------------------------------------
    def draw_result(self, frame: np.ndarray,
                    result: ColorCheckResult,
                    body_box: Optional[Tuple] = None,
                    position: Tuple = (10, 90)) -> np.ndarray:
        """
        Ve ket qua phan tich mau len frame.
        Goi sau khi da ve ket qua TFLite (uniform_checker.draw_result).
        """
        COLOR_MAP = {
            "OK"     : (0, 200, 50),
            "FAIL"   : (0, 50, 220),
            "SKIP"   : (180, 120, 0),
            "UNCLEAR": (0, 165, 255),
        }
        color = COLOR_MAP.get(result.status, (200, 200, 200))
        x, y  = position

        # Ve duong phan vung ao/quan de debug (neu co body_box)
        if body_box is not None:
            bx, by, bw, bh = body_box
            # Vung ao (xanh la nhat)
            shirt_y0 = by + int(bh * self.SHIRT_ZONE[0])
            shirt_y1 = by + int(bh * self.SHIRT_ZONE[1])
            cv2.rectangle(frame, (bx, shirt_y0), (bx+bw, shirt_y1),
                          (0, 200, 100), 1)
            cv2.putText(frame, "Ao", (bx + 5, shirt_y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1)

            # Vung quan (xanh duong nhat)
            pants_y0 = by + int(bh * self.PANTS_ZONE[0])
            pants_y1 = by + int(bh * self.PANTS_ZONE[1])
            cv2.rectangle(frame, (bx, pants_y0), (bx+bw, pants_y1),
                          (200, 100, 0), 1)
            cv2.putText(frame, "Quan", (bx + 5, pants_y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 0), 1)

        # Nen mo cho text
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - 5), (x + 440, y + 80),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Dong chu ket qua chinh
        cv2.putText(frame, f"[Mau sac] {result.message}",
                    (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

        # Dong chi tiet ao
        shirt_pct = result.shirt_ratio * 100
        shirt_icon = "V" if result.is_shirt_ok else "X"
        cv2.putText(frame,
                    f"  Ao: {result.shirt_color} ({shirt_pct:.0f}%) [{shirt_icon}]",
                    (x, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 220, 60) if result.is_shirt_ok else (0, 80, 220), 1)

        # Dong chi tiet quan
        pants_pct = result.pants_dark_ratio * 100
        pants_icon = "V" if result.is_pants_ok else "X"
        cv2.putText(frame,
                    f"  Quan: {result.pants_color} (toi: {pants_pct:.0f}%) [{pants_icon}]",
                    (x, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 220, 60) if result.is_pants_ok else (0, 80, 220), 1)

        return frame


# ==============================================================
# TEST TRUC TIEP - CHAY: python uniform_color_analyzer.py
# ==============================================================
def run_live_debug():
    """Test phan tich mau sac realtime voi thanh debug HSV."""
    import sys

    analyzer = UniformColorAnalyzer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera!")
        return

    # Cho phep chon ngay de test
    print("\nChon ngay de test (mac dinh = hom nay):")
    print("  0=Thu2  1=Thu3  2=Thu4  3=Thu5  4=Thu6  5=Thu7  6=CN")
    inp = input("Nhap so (Enter de dung ngay hom nay): ").strip()
    if inp.isdigit():
        forced_weekday = int(inp)
    else:
        forced_weekday = None

    day_names = ["Thu 2", "Thu 3", "Thu 4", "Thu 5", "Thu 6", "Thu 7", "CN"]
    wday = forced_weekday if forced_weekday is not None else datetime.now().weekday()
    print(f"[INFO] Dang test ngay: {day_names[wday]} | Nhan Q de thoat\n")

    show_debug = False  # Nhan D de bat/tat debug mask

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        result = analyzer.check(frame, weekday=forced_weekday)
        frame  = analyzer.draw_result(frame, result)

        # Debug mask (bat/tat bang phim D)
        if show_debug:
            h_frame, w_frame = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            white_mask = cv2.inRange(hsv,
                np.array([0,   0,         WHITE_V_MIN]),
                np.array([179, WHITE_S_MAX, 255]))
            blue_mask = cv2.inRange(hsv,
                np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]),
                np.array([BLUE_H_MAX, 255, 255]))
            dark_mask = cv2.inRange(hsv,
                np.array([0, 0, 0]),
                np.array([179, 255, DARK_V_MAX]))

            # Hien thi mask nho goc phai
            thumb_w = 150
            for i, (mask, label, color) in enumerate([
                (white_mask, "Trang", (220, 220, 220)),
                (blue_mask,  "Xanh",  (220, 100, 0)),
                (dark_mask,  "Toi",   (60,  60,  60)),
            ]):
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                resized  = cv2.resize(mask_rgb, (thumb_w, thumb_w))
                x_off    = w_frame - thumb_w - 10
                y_off    = 10 + i * (thumb_w + 5)
                frame[y_off:y_off+thumb_w, x_off:x_off+thumb_w] = resized
                cv2.putText(frame, label, (x_off, y_off - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Huong dan phim
        cv2.putText(frame, "D=debug  Q=thoat",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Thong tin debug raw
        d = result.debug_colors
        cv2.putText(frame,
                    f"HSV ao: {d['shirt_hsv_mean']} | "
                    f"W:{d['white_ratio']:.2f} B:{d['blue_ratio']:.2f} "
                    f"Dk:{d['dark_pants_ratio']:.2f}",
                    (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imshow("Kiem tra mau sac trang phuc - Debug", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[DEBUG] Hien thi mask: {'BAT' if show_debug else 'TAT'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        img_path = sys.argv[idx + 1]
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] Khong doc duoc anh: {img_path}")
            sys.exit(1)

        inp = input("Nhap so thu (0=T2, 1=T3, 2=T4, 3=T5, 4=T6): ").strip()
        wday = int(inp) if inp.isdigit() else None

        analyzer = UniformColorAnalyzer()
        result = analyzer.check(frame, weekday=wday)

        print("\n===== KET QUA PHAN TICH MAU SAC =====")
        print(f"  Trang thai     : {result.status}")
        print(f"  Ao             : {result.shirt_color} ({result.shirt_ratio*100:.1f}%)")
        print(f"  Quan           : {result.pants_color} (toi: {result.pants_dark_ratio*100:.1f}%)")
        print(f"  Ao dung quy dinh: {'Co' if result.is_shirt_ok else 'Khong'}")
        print(f"  Quan toi mau   : {'Co' if result.is_pants_ok else 'Khong'}")
        print(f"  Thong bao      : {result.message}")
        print(f"  Debug          : {result.debug_colors}")

        frame = analyzer.draw_result(frame, result)
        cv2.imshow("Ket qua phan tich mau", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        run_live_debug()
