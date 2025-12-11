# Đo trung bình 10 lần

import cv2
import pyrealsense2 as rs
import numpy as np
from collections import deque

# === Khởi tạo pipeline ===
pipeline = rs.pipeline()
config = rs.config()

# Bật stream màu (RGB) và stream độ sâu
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Bắt đầu stream
pipeline.start(config)

# Lưu điểm click
points = []

# Bộ nhớ lưu 10 giá trị đo gần nhất cho từng điểm
buffers = [deque(maxlen=20), deque(maxlen=20)]  # tối đa 2 điểm

# === Hàm callback chuột ===
def click(event, x, y, flags, param):
    global points, buffers
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y)))  # ép kiểu int để tránh lỗi numpy.int64
        if len(points) > 2:
            points.pop(0)
        # Reset bộ đệm mỗi khi chọn điểm mới
        buffers = [deque(maxlen=10), deque(maxlen=10)]
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()
        buffers = [deque(maxlen=50), deque(maxlen=50)]

cv2.namedWindow("Bgr frame")
cv2.setMouseCallback("Bgr frame", click)

while True:
    # Lấy frame mới
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Chuyển sang numpy array
    bgr_frame = np.asanyarray(color_frame.get_data())

    # Hiển thị các điểm và khoảng cách trung bình (đơn vị mm)
    for i, (x, y) in enumerate(points):
        distance = depth_frame.get_distance(x, y) * 1000  # mét → mm

        # Lưu vào bộ đệm tương ứng
        if i < len(buffers):
            buffers[i].append(distance)

        # Nếu có dữ liệu, lấy trung bình
        if len(buffers[i]) > 0:
            avg_distance = int(np.mean(buffers[i]))  # ép int => bỏ .0
        else:
            avg_distance = int(distance)

        # Vẽ lên ảnh
        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        cv2.circle(bgr_frame, (x, y), 8, color, -1)
        cv2.putText(bgr_frame, f"{avg_distance}mm",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC để thoát
        break

pipeline.stop()
cv2.destroyAllWindows()
