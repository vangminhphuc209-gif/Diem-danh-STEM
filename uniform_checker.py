# -*- coding: utf-8 -*-
"""
=============================================================
  BUOC 3: MODULE NHAN DIEN TRANG PHUC REALTIME
  uniform_checker.py

  Chức năng:
  - Nhận diện trang phục học sinh qua camera realtime
  - Tự động biết hôm nay là thứ mấy → quy định trang phục gì
  - Thứ 7, Chủ nhật: Không yêu cầu trang phục
  - Trả về: Đúng / Sai / Không yêu cầu
  
  Dùng:
    Import vào main.py  : from uniform_checker import UniformChecker
    Chạy test camera    : python uniform_checker.py --test
    Chạy test ảnh file  : python uniform_checker.py --image path/to/img.jpg
=============================================================
"""

import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from enum import Enum

# ==================== CẤU HÌNH ====================
BASE_DIR      = Path(__file__).parent
MODEL_PATH    = BASE_DIR / "data" / "uniform_model.tflite"
LABELS_PATH   = BASE_DIR / "data" / "uniform_labels.json"
IMAGE_SIZE    = (224, 224)
CONFIDENCE_THRESHOLD = 0.55  # Độ tự tin tối thiểu, thấp hơn = "Không rõ"


# ==================== QUY ĐỊNH TRANG PHỤC THEO NGÀY ====================
class UniformRule(Enum):
    TRANG_PHUC_DAN_TOC = "trang_phuc_dan_toc"  # Thứ 2
    AO_TRANG           = "ao_trang"             # Thứ 3, Thứ 5
    TU_DO_CO_CO        = "tu_do_co_co"          # Thứ 4 (không phân biệt bằng AI, dùng logic)
    AO_DOAN            = "ao_doan"              # Thứ 6
    NGHI               = "nghi"                 # Thứ 7, Chủ nhật


# weekday(): 0=Thứ 2, 1=Thứ 3, 2=Thứ 4, 3=Thứ 5, 4=Thứ 6, 5=Thứ 7, 6=CN
WEEKDAY_RULES = {
    0: UniformRule.TRANG_PHUC_DAN_TOC,
    1: UniformRule.AO_TRANG,
    2: UniformRule.TU_DO_CO_CO,
    3: UniformRule.AO_TRANG,
    4: UniformRule.AO_DOAN,
    5: UniformRule.NGHI,
    6: UniformRule.NGHI,
}

# Hiển thị tiếng Việt
RULE_DISPLAY = {
    UniformRule.TRANG_PHUC_DAN_TOC : "Trang phục dân tộc",
    UniformRule.AO_TRANG           : "Áo trắng",
    UniformRule.TU_DO_CO_CO        : "Tự do (phải có cổ)",
    UniformRule.AO_DOAN            : "Áo đoàn viên",
    UniformRule.NGHI               : "Nghỉ - Không yêu cầu",
}

# Màu hiển thị trên camera (BGR)
COLOR_OK      = (0, 200, 50)    # Xanh lá
COLOR_FAIL    = (0, 50, 220)    # Đỏ
COLOR_SKIP    = (180, 120, 0)   # Xanh dương (ngày tự do)
COLOR_UNCLEAR = (0, 165, 255)   # Cam (không chắc chắn)


class UniformChecker:
    """Module nhận diện trang phục tích hợp vào hệ thống điểm danh."""

    def __init__(self):
        self.interpreter  = None
        self.labels       = []
        self.input_details  = None
        self.output_details = None
        self._load_model()

    def _load_model(self):
        """Nạp model TFLite vào bộ nhớ."""
        if not MODEL_PATH.exists():
            print(f"[WARN] Chưa có model: {MODEL_PATH}")
            print("       Hãy chạy: python uniform_trainer.py")
            return

        if not LABELS_PATH.exists():
            print(f"[WARN] Chưa có file nhãn: {LABELS_PATH}")
            return

        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        self.interpreter = Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(LABELS_PATH, encoding="utf-8") as f:
            self.labels = json.load(f)

        print(f"[OK] Đã nạp model nhận diện trang phục ({len(self.labels)} nhãn)")
        print(f"     Nhãn: {self.labels}")

    def is_model_ready(self) -> bool:
        return self.interpreter is not None

    # ------------------------------------------------------------------
    # LOGIC QUY ĐỊNH THEO NGÀY
    # ------------------------------------------------------------------
    @staticmethod
    def get_today_rule(weekday: int = None) -> UniformRule:
        """Trả về quy định trang phục hôm nay."""
        if weekday is None:
            weekday = datetime.now().weekday()
        return WEEKDAY_RULES.get(weekday, UniformRule.NGHI)

    @staticmethod
    def get_today_rule_display(weekday: int = None) -> str:
        rule = UniformChecker.get_today_rule(weekday)
        return RULE_DISPLAY[rule]

    # ------------------------------------------------------------------
    # NHẬN DIỆN TRANG PHỤC
    # ------------------------------------------------------------------
    def predict(self, frame: np.ndarray, body_box: tuple = None) -> dict:
        """
        Nhan dien trang phuc trong anh.
        - Thu 2: Dung TFLite AI (trang phuc dan toc)
        - Thu 3, 5: Dung Color Analyzer (ao trang + quan toi)
        - Thu 4: SKIP (tu do co co - khong check tu dong)
        - Thu 6: Dung Color Analyzer (ao xanh + quan toi)
        - Thu 7, CN: SKIP (nghi)
        """
        from uniform_color_analyzer import UniformColorAnalyzer

        today_rule = self.get_today_rule()
        rule_text  = RULE_DISPLAY[today_rule]
        weekday    = datetime.now().weekday()

        # ---- NGAY NGHI -> SKIP ----
        if today_rule == UniformRule.NGHI:
            return self._result("nghi", 1.0, True, rule_text, "SKIP",
                                "Cuoi tuan - Khong yeu cau trang phuc")

        # ---- THU 4: TU DO CO CO -> SKIP ----
        if today_rule == UniformRule.TU_DO_CO_CO:
            return self._result("tu_do", 1.0, True, rule_text, "SKIP",
                                "Thu 4: Tu do - Khong kiem tra tu dong")

        # ---- THU 3, 5, 6: DUNG COLOR ANALYZER ----
        if today_rule in (UniformRule.AO_TRANG, UniformRule.AO_DOAN):
            color_analyzer = UniformColorAnalyzer()
            color_result   = color_analyzer.check(frame, body_box, weekday)

            is_correct = color_result.status == "OK"
            status     = color_result.status
            msg        = color_result.message

            return {
                "label"        : color_result.shirt_color,
                "confidence"   : color_result.shirt_ratio,
                "is_correct"   : is_correct,
                "rule"         : rule_text,
                "status"       : status,
                "message"      : msg,
                "color_result" : color_result,   # Du lieu goc de draw_result dung
            }

        # ---- THU 2: DUNG TFLITE AI (trang phuc dan toc) ----
        if not self.is_model_ready():
            return self._result("unknown", 0.0, False, rule_text, "NO_MODEL",
                                "Model chua duoc training")

        # Cat vung than nguoi
        if body_box is not None:
            x, y, w, h = body_box
            y_start = max(0, y + int(h * 0.15))
            y_end   = min(frame.shape[0], y + int(h * 0.80))
            region  = frame[y_start:y_end, x:x + w]
        else:
            h, w    = frame.shape[:2]
            y_start = h // 4
            y_end   = h * 3 // 4
            region  = frame[y_start:y_end, :]

        if region.size == 0:
            region = frame

        img = cv2.resize(region, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        predicted_idx   = int(np.argmax(output))
        confidence      = float(np.max(output))
        predicted_label = self.labels[predicted_idx] if predicted_idx < len(self.labels) else "unknown"

        if confidence < CONFIDENCE_THRESHOLD:
            return self._result(predicted_label, confidence, False, rule_text, "UNCLEAR",
                                f"Khong ro trang phuc ({confidence*100:.0f}%)")

        is_correct = self._check_rule(predicted_label, today_rule)
        if is_correct:
            msg    = f"OK: Trang phuc dan toc ({confidence*100:.0f}%)"
            status = "OK"
        else:
            msg    = f"SAI: Can mac trang phuc dan toc ({confidence*100:.0f}%)"
            status = "FAIL"

        return self._result(predicted_label, confidence, is_correct,
                            rule_text, status, msg)


    def _check_rule(self, predicted_label: str, rule: UniformRule) -> bool:
        """Kiểm tra nhãn dự đoán có khớp với quy định hôm nay không."""
        mapping = {
            UniformRule.TRANG_PHUC_DAN_TOC : ["trang_phuc_dan_toc"],
            UniformRule.AO_TRANG           : ["ao_trang"],
            UniformRule.AO_DOAN            : ["ao_doan"],
        }
        allowed = mapping.get(rule, [])
        return predicted_label in allowed

    @staticmethod
    def _result(label, conf, correct, rule, status, msg) -> dict:
        return {
            "label"      : label,
            "confidence" : round(conf, 3),
            "is_correct" : correct,
            "rule"       : rule,
            "status"     : status,
            "message"    : msg,
        }

    # ------------------------------------------------------------------
    # VẼ KẾT QUẢ LÊN FRAME (Dùng trong chế độ realtime)
    # ------------------------------------------------------------------
    def draw_result(self, frame: np.ndarray, result: dict,
                    position: tuple = (10, 30)) -> np.ndarray:
        """Vẽ thông tin trang phục lên frame camera."""
        status  = result.get("status", "")
        message = result.get("message", "")
        rule    = result.get("rule", "")
        
        # Chọn màu theo trạng thái
        color_map = {
            "OK"      : COLOR_OK,
            "FAIL"    : COLOR_FAIL,
            "SKIP"    : COLOR_SKIP,
            "UNCLEAR" : COLOR_UNCLEAR,
            "NO_MODEL": COLOR_UNCLEAR,
        }
        color = color_map.get(status, COLOR_UNCLEAR)

        x, y = position
        # Nền mờ cho chữ
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - 20), (x + 420, y + 55), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Dòng 1: Quy định hôm nay
        weekday_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]
        today_name    = weekday_names[datetime.now().weekday()]
        cv2.putText(frame, f"{today_name}: {rule}",
                    (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Dòng 2: Kết quả nhận diện
        cv2.putText(frame, message,
                    (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return frame


# ==============================================================
# CHẠY TEST TRỰC TIẾP
# ==============================================================
def run_camera_test():
    """Test nhận diện trang phục realtime qua webcam."""
    checker = UniformChecker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Không mở được camera!")
        return

    print("[INFO] Đang mở camera test... Nhấn 'Q' để thoát")
    print(f"[INFO] Hôm nay: {checker.get_today_rule_display()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Lật gương

        # Nhận diện trang phục
        result = checker.predict(frame)

        # Vẽ kết quả lên frame
        frame = checker.draw_result(frame, result)

        # Hiển thị FPS và thời gian
        now = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, now, (frame.shape[1] - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Kiem tra trang phuc hoc sinh - Nhan Q de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image_test(image_path: str):
    """Test nhận diện trang phục trên một ảnh tĩnh."""
    checker = UniformChecker()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Không đọc được ảnh: {image_path}")
        return
    
    result = checker.predict(frame)
    
    print("\n===== KẾT QUẢ NHẬN DIỆN =====")
    for k, v in result.items():
        print(f"  {k:15s}: {v}")
    
    # Vẽ và hiển thị
    frame = checker.draw_result(frame, result)
    
    # Resize nếu ảnh quá to
    h, w = frame.shape[:2]
    if w > 900:
        scale = 900 / w
        frame = cv2.resize(frame, (900, int(h * scale)))
    
    cv2.imshow("Kiem tra trang phuc - Nhan phim bat ky de thoat", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if "--image" in args:
        idx = args.index("--image")
        if idx + 1 < len(args):
            run_image_test(args[idx + 1])
        else:
            print("Dùng: python uniform_checker.py --image <duong_dan_anh>")
    else:
        # Mặc định: test camera
        run_camera_test()
