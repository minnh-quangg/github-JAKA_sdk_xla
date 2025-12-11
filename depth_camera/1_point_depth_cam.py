import cv2
import numpy as np
import pyrealsense2 as rs

# --- Cấu hình & khởi động camera ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Lấy hệ số đổi đơn vị depth (mỗi value z16 * depth_scale -> mét)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # mét / đơn vị
# Canh hàng: đưa depth “khớp” với khung màu
align = rs.align(rs.stream.color)

# (Tùy chọn) Lọc lỗ để depth mượt hơn
hole_filling = rs.hole_filling_filter()

# Biến toàn cục để lưu điểm click
point = (320, 240)

def show_distance(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

cv2.namedWindow("Color frame", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Color frame", show_distance)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # (Tùy chọn) lọc lỗ
        depth_frame = hole_filling.process(depth_frame)

        # Convert sang numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Lấy khoảng cách tại pixel đã chọn
        x, y = point
        h, w = depth_image.shape
        if 0 <= x < w and 0 <= y < h:
            raw = depth_image[y, x]  # đơn vị: z16 (không phải mm)
            distance_m = float(raw) * depth_scale  # mét
            # Một số pixel có thể = 0 nếu out-of-range / noise
            if distance_m <= 0:
                text = "No depth"
            else:
                text = f"{distance_m*1000:.0f} mm"
        else:
            text = "Out of frame"

        # Vẽ điểm & text
        cv2.circle(color_image, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(color_image, text, (x + 8, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
        cv2.imshow("Color frame", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC hoặc q để thoát
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
