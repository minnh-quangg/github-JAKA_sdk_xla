from ultralytics import YOLO
import cv2
from realsense_camera import *

# Khởi tạo camera và model
rs = RealsenseCamera()
model = YOLO("best.pt")
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

while True:
    ret, bgr, _ = rs.get_frame_stream()  # Bỏ depth
    if not ret:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    frame = bgr.copy()
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)

        # Tính tâm của bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Vẽ bbox, label và tâm
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Chấm tâm đỏ
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cv2.destroyAllWindows()
