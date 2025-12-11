import pyrealsense2 as rs
import numpy as np
import cv2

# ==== 1. Cấu hình pipeline ====
pipeline = rs.pipeline()
config = rs.config()

# Chọn luồng RGB (color stream)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ==== 2. Bắt đầu pipeline ====
pipeline.start(config)

print("Đang mở camera Intel RealSense D435i...")

try:
    while True:
        # ==== 3. Lấy frame từ camera ====
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # ==== 4. Chuyển frame sang dạng numpy (OpenCV) ====
        color_image = np.asanyarray(color_frame.get_data())

        # ==== 5. Hiển thị bằng OpenCV ====
        cv2.imshow('RGB Camera (D435i)', color_image)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # ==== 6. Giải phóng tài nguyên ====
    pipeline.stop()
    cv2.destroyAllWindows()
