# -*- coding: utf-8 -*-
"""
=============================================================
  MODULE PHÁT HIỆN NGƯỜI — HOG PERSON DETECTOR
  person_detector.py

  Tích hợp 2 phương pháp phát hiện người:
  1. HOG + SVM (OpenCV built-in) — Chính xác, không cần GPU
  2. Face-based fallback — Dùng khi HOG không phát hiện được

  Cách dùng:
    from person_detector import PersonDetector
    detector = PersonDetector()
    boxes = detector.detect(frame)   # List[dict]
    # boxes[i] = {
    #   "body_box"  : (x, y, w, h),   # Vùng thân người (đầy đủ)
    #   "shirt_box" : (x, y, w, h),   # Vùng áo (15%–55% chiều cao)
    #   "pants_box" : (x, y, w, h),   # Vùng quần (55%–90% chiều cao)
    #   "source"    : "hog" | "face"  # Nguồn phát hiện
    # }
=============================================================
"""

import cv2
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# CẤU HÌNH HOG DETECTOR
# ──────────────────────────────────────────────────────────────────────

# Kích thước cửa sổ phát hiện (yêu cầu bởi HOG SVM OpenCV)
HOG_WIN_STRIDE  = (8, 8)
HOG_PADDING     = (4, 4)
HOG_SCALE       = 1.05      # Nhỏ hơn = tìm nhiều hơn, chậm hơn
HOG_FINAL_THRESH= 0         # Ngưỡng confidence HOG (nâng lên nếu quá nhiều false positive)

# Scale frame trước khi detect để tăng tốc (0.5 = 50% kích thước gốc)
DETECT_SCALE    = 0.6

# Ngưỡng NMS (Non-Maximum Suppression) — lọc bỏ hộp chồng nhau
NMS_OVERLAP_THRESH = 0.65

# Vùng phân tích áo/quần (tính theo tỷ lệ chiều cao body_box)
SHIRT_ZONE_TOP    = 0.12   # Tránh đầu/cổ
SHIRT_ZONE_BOTTOM = 0.52   # Điểm cắt áo-quần
PANTS_ZONE_TOP    = 0.52
PANTS_ZONE_BOTTOM = 0.90   # Tránh giày dép

# Face Cascade (fallback)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Ước tính thân người từ mặt:
# Thân người trung bình cao gấp ~7 lần chiều cao khuôn mặt
# Mặt chiếm khoảng phần trên 15% chiều cao cơ thể
FACE_TO_BODY_HEIGHT_RATIO = 7.0
FACE_TO_BODY_WIDTH_RATIO  = 1.8


# ──────────────────────────────────────────────────────────────────────
# CLASS CHÍNH
# ──────────────────────────────────────────────────────────────────────
class PersonDetector:
    """
    Phát hiện người trong frame camera và tính toán vùng áo/quần.

    Ưu tiên HOG (thân đầy đủ), fallback sang face-based estimation.
    """

    def __init__(self, use_face_fallback: bool = True, verbose: bool = False):
        self.verbose         = verbose
        self.use_face_fallback = use_face_fallback

        # ── Khởi tạo HOG ──
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        if verbose:
            print("[PersonDetector] HOG SVM người đã sẵn sàng.")

        # ── Khởi tạo Face Cascade (fallback) ──
        self.face_cascade = None
        if use_face_fallback:
            try:
                self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
                if self.face_cascade.empty():
                    print("[PersonDetector] WARN: Không tải được face cascade!")
                    self.face_cascade = None
                elif verbose:
                    print("[PersonDetector] Face cascade đã sẵn sàng (fallback).")
            except Exception as e:
                print(f"[PersonDetector] WARN: {e}")

    # ──────────────────────────────────────────────────────────────────
    # HÀM PHÁT HIỆN CHÍNH
    # ──────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray,
               min_person_height: int = 80) -> list[dict]:
        """
        Phát hiện người trong frame và trả về danh sách bounding box.

        Args:
            frame             : Frame BGR từ camera.
            min_person_height : Chiều cao tối thiểu (px) để coi là người.

        Returns:
            List[dict] với mỗi phần tử là:
            {
                "body_box"  : (x, y, w, h),   # Vùng thân người
                "shirt_box" : (x, y, w, h),   # Vùng áo
                "pants_box" : (x, y, w, h),   # Vùng quần
                "source"    : "hog" | "face"  # Nguồn phát hiện
            }
        """
        h_frame, w_frame = frame.shape[:2]
        results = []

        # ── Thử HOG trước ──
        hog_boxes = self._detect_hog(frame, min_person_height)

        if hog_boxes:
            for box in hog_boxes:
                parsed = self._parse_body_box(box, w_frame, h_frame)
                parsed["source"] = "hog"
                results.append(parsed)
            if self.verbose:
                print(f"[PersonDetector] HOG: {len(results)} người phát hiện.")
            return results

        # ── Fallback: Face-based ──
        if self.use_face_fallback and self.face_cascade is not None:
            face_boxes = self._detect_faces(frame)
            if face_boxes:
                for fbox in face_boxes:
                    body_box = self._estimate_body_from_face(fbox, w_frame, h_frame)
                    parsed   = self._parse_body_box(body_box, w_frame, h_frame)
                    parsed["source"] = "face"
                    results.append(parsed)
                if self.verbose:
                    print(f"[PersonDetector] FALLBACK Face→Body: {len(results)} người.")
                return results

        # ── Không tìm thấy gì → Dùng toàn frame ──
        if self.verbose:
            print("[PersonDetector] Không phát hiện người — dùng toàn frame.")
        full_box = (0, 0, w_frame, h_frame)
        parsed   = self._parse_body_box(full_box, w_frame, h_frame)
        parsed["source"] = "full_frame"
        return [parsed]

    # ──────────────────────────────────────────────────────────────────
    # HOG DETECTION
    # ──────────────────────────────────────────────────────────────────
    def _detect_hog(self, frame: np.ndarray,
                    min_height: int) -> list[tuple]:
        """Chạy HOG detector và trả về danh sách (x,y,w,h) sau NMS."""
        # Scale nhỏ để tăng tốc
        h, w    = frame.shape[:2]
        small   = cv2.resize(frame, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))

        try:
            rects, weights = self.hog.detectMultiScale(
                small,
                winStride    = HOG_WIN_STRIDE,
                padding      = HOG_PADDING,
                scale        = HOG_SCALE,
                finalThreshold = HOG_FINAL_THRESH
            )
        except cv2.error:
            return []

        if len(rects) == 0:
            return []

        # Scale box trở về kích thước gốc
        inv = 1.0 / DETECT_SCALE
        rects_orig = [
            (int(x * inv), int(y * inv), int(w2 * inv), int(h2 * inv))
            for (x, y, w2, h2) in rects
        ]

        # Lọc theo chiều cao tối thiểu
        rects_orig = [(x, y, w2, h2) for (x, y, w2, h2) in rects_orig if h2 >= min_height]
        if not rects_orig:
            return []

        # NMS — loại bỏ hộp chồng nhau
        return self._nms(rects_orig, NMS_OVERLAP_THRESH)

    # ──────────────────────────────────────────────────────────────────
    # FACE DETECTION (FALLBACK)
    # ──────────────────────────────────────────────────────────────────
    def _detect_faces(self, frame: np.ndarray) -> list[tuple]:
        """Phát hiện khuôn mặt trong frame."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)   # Cải thiện độ tương phản
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors= 5,
            minSize     = (30, 30)
        )
        return list(faces) if len(faces) > 0 else []

    def _estimate_body_from_face(self, face_box: tuple,
                                  frame_w: int, frame_h: int) -> tuple:
        """
        Ước tính bounding box thân người từ khuôn mặt.

        Nguyên lý: Trung bình đầu người = 1/7.5 chiều cao cơ thể.
        Đặt đầu vào 1/7 trên cùng → tính thân xuống dưới.
        """
        fx, fy, fw, fh = face_box

        # Mở rộng chiều ngang để bao thân
        body_w = int(fw * FACE_TO_BODY_WIDTH_RATIO)
        body_h = int(fh * FACE_TO_BODY_HEIGHT_RATIO)

        # Căn giữa theo trục ngang
        body_x = fx + fw // 2 - body_w // 2
        body_y = fy - int(fh * 0.3)  # Một chút phía trên khuôn mặt

        # Clip vào frame
        body_x = max(0, body_x)
        body_y = max(0, body_y)
        body_w = min(body_w, frame_w - body_x)
        body_h = min(body_h, frame_h - body_y)

        return (body_x, body_y, body_w, body_h)

    # ──────────────────────────────────────────────────────────────────
    # TINH CHỈNH VÙNG ÁO / QUẦN
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_body_box(body_box: tuple,
                         frame_w: int, frame_h: int) -> dict:
        """
        Từ body_box (x,y,w,h), tính vùng áo và quần theo tỷ lệ.
        """
        x, y, w, h = body_box

        # Clip body box vào frame
        x = max(0, x); y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        # Vùng áo
        sy0 = y + int(h * SHIRT_ZONE_TOP)
        sy1 = y + int(h * SHIRT_ZONE_BOTTOM)
        sy0 = max(0, min(sy0, frame_h - 1))
        sy1 = max(sy0 + 1, min(sy1, frame_h))

        # Vùng quần
        py0 = y + int(h * PANTS_ZONE_TOP)
        py1 = y + int(h * PANTS_ZONE_BOTTOM)
        py0 = max(0, min(py0, frame_h - 1))
        py1 = max(py0 + 1, min(py1, frame_h))

        return {
            "body_box"  : (x, y, w, h),
            "shirt_box" : (x, sy0, w, sy1 - sy0),
            "pants_box" : (x, py0, w, py1 - py0),
            "source"    : "unknown",
        }

    # ──────────────────────────────────────────────────────────────────
    # VẼ BOUNDING BOX LÊN FRAME (Debug / UI)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def draw_detections(frame: np.ndarray,
                        detections: list[dict],
                        draw_zones: bool = True) -> np.ndarray:
        """
        Vẽ bounding box và vùng áo/quần lên frame để kiểm tra.

        Args:
            frame      : Frame BGR gốc (sẽ được vẽ lên).
            detections : Kết quả từ detect().
            draw_zones : True = vẽ thêm vùng áo (xanh lá) và quần (xanh dương).

        Returns:
            Frame đã vẽ.
        """
        SOURCE_COLOR = {
            "hog"        : (0, 255, 0),      # Xanh lá — HOG chính xác
            "face"       : (255, 165, 0),    # Cam — Ước tính từ mặt
            "full_frame" : (150, 150, 150),  # Xám — Không phát hiện được
        }
        SHIRT_COLOR = (50, 220, 50)    # Xanh lá nhạt
        PANTS_COLOR = (220, 100, 50)   # Cam nhạt

        for d in detections:
            src   = d.get("source", "unknown")
            color = SOURCE_COLOR.get(src, (200, 200, 200))
            x, y, w, h = d["body_box"]

            # Vẽ body box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Person [{src}]",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if draw_zones:
                # Vùng áo
                sx, sy, sw, sh = d["shirt_box"]
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), SHIRT_COLOR, 1)
                cv2.putText(frame, "Ao", (sx + 4, sy + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, SHIRT_COLOR, 1)

                # Vùng quần
                px, py, pw, ph = d["pants_box"]
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), PANTS_COLOR, 1)
                cv2.putText(frame, "Quan", (px + 4, py + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANTS_COLOR, 1)

        return frame

    # ──────────────────────────────────────────────────────────────────
    # NMS — NON-MAXIMUM SUPPRESSION
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _nms(boxes: list[tuple], overlap_thresh: float) -> list[tuple]:
        """
        Áp dụng Non-Maximum Suppression để loại bỏ hộp chồng nhau.
        Dựa trên thuật toán Malisiewicz et al. (fast NMS).
        """
        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=float)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
        y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sắp xếp theo y2 (dưới cùng) — chọn hộp thấp nhất (thân người đầy đủ hơn)
        idxs = np.argsort(y2)
        pick = []

        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)

            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])

            inter_w = np.maximum(0, xx2 - xx1 + 1)
            inter_h = np.maximum(0, yy2 - yy1 + 1)
            overlap  = (inter_w * inter_h) / areas[idxs[:-1]]

            idxs = np.delete(idxs, np.concatenate([[len(idxs) - 1],
                                                    np.where(overlap > overlap_thresh)[0]]))

        return [tuple(int(v) for v in boxes_arr[i]) for i in pick]


# ──────────────────────────────────────────────────────────────────────
# TEST TRỰC TIẾP QUA CAMERA
# ──────────────────────────────────────────────────────────────────────
def _run_camera_test():
    """Test PersonDetector realtime qua webcam. Nhấn Q để thoát, Z để bật/tắt zones."""
    detector = PersonDetector(verbose=True)
    cap      = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Không mở được camera!")
        return

    print("=== TEST PERSON DETECTOR ===")
    print("  HOG phát hiện người → vẽ body box + vùng áo/quần")
    print("  Q = Thoát | Z = Bật/tắt zone visualization")

    show_zones = True
    frame_count = 0
    fps_start   = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Phát hiện người
        detections = detector.detect(frame)

        # Vẽ kết quả
        frame = PersonDetector.draw_detections(frame, detections, draw_zones=show_zones)

        # FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
            fps = frame_count / elapsed
            frame_count = 0
            fps_start   = cv2.getTickCount()
        else:
            fps = 0

        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}  Nguoi: {len(detections)}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, "Q=Thoat  Z=Zones",
                    (frame.shape[1] - 160, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Person Detector Test — HOG", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("z"):
            show_zones = not show_zones
            print(f"[TEST] Zones: {'BẬT' if show_zones else 'TẮT'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_camera_test()
