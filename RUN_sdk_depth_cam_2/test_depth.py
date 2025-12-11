import pyrealsense2 as rs
import numpy as np
import cv2

# === Khởi tạo pipeline ===
pipeline = rs.pipeline()
config = rs.config()

# Bật 2 stream: color + depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# === Bắt đầu stream ===
pipeline.start(config)

try:
    while True:
        # Chờ frame mới từ camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Chuyển frame sang numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Ánh xạ màu cho ảnh depth (để hiển thị dễ nhìn)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )

        # Gộp 2 ảnh song song
        images = np.hstack((color_image, depth_colormap))

        # Hiển thị
        cv2.imshow('RealSense Color + Depth', images)

        # Nhấn ESC để thoát
        if cv2.waitKey(1) == 27:
            break

finally:
    # Dừng pipeline và đóng cửa sổ
    pipeline.stop()
    cv2.destroyAllWindows()
