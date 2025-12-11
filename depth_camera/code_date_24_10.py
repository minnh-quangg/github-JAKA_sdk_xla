from ultralytics import YOLO
import cv2
import numpy as np
from realsense_camera import *

rs = RealsenseCamera()
model = YOLO("best.pt")
cv2.namedWindow("Measure Distance (mm)", cv2.WINDOW_NORMAL)

# Hàm lọc trung vị (median) cho ma trận kxk
def median_depth_mm(depth_img, cx, cy, k=5):
    h, w = depth_img.shape[:2]
    r = k // 2
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r + 1, w) #chỉ lấy đến end-1
    y2 = min(cy + r + 1, h)
    roi = depth_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0
    return int(np.median(valid))

while True:
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    if not ret:
        break

    frame = bgr_frame.copy()
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0]) #trái: tym-0, tròn-1, vuông-2
        conf = float(box.conf[0])
        label = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        dist_mm = median_depth_mm(depth_frame, cx, cy, k=5)
        delta_mm = 264 - dist_mm
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Nhãn
        txt1 = f"{label} {conf:.2f}" #độ tin cậy
        # txt2 = f"{dist_mm}mm" if dist_mm > 0 else "N/A" #khoảng cách đo
        txt3 = f"Distance = {delta_mm}mm" if dist_mm > 0 else "Distance = N/A" #độ cao vật

        # Chọn vị trí hiển thị “ngoài” bbox (ưu tiên phía trên, nếu không đủ chỗ thì bên dưới)
        pad = 8
        text_y_top = y1 - 10
        text_x = x1
        if text_y_top > 25:
            # Hiển thị trên bbox
            cv2.putText(frame, txt1, (text_x, text_y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt1, (text_x, text_y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # cv2.putText(frame, txt2, (text_x, text_y_top - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            # cv2.putText(frame, txt2, (text_x, text_y_top - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 50), 2, cv2.LINE_AA)

            cv2.putText(frame, txt3, (text_x, text_y_top - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt3, (text_x, text_y_top - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
        else:
            # Hiển thị dưới bbox
            base_y = y2 + 20
            cv2.putText(frame, txt1, (text_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt1, (text_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # cv2.putText(frame, txt2, (text_x, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            # cv2.putText(frame, txt2, (text_x, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 50), 2, cv2.LINE_AA)

            cv2.putText(frame, txt3, (text_x, base_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt3, (text_x, base_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)

    cv2.imshow("Measure Distance (mm)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
