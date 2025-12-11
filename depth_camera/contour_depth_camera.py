import pyrealsense2 as rs
import numpy as np
import cv2

# Khởi tạo pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Bắt đầu pipeline
pipeline.start(config)

# Bộ lọc để làm mượt ảnh depth
dec_filter = rs.decimation_filter()
spat_filter = rs.spatial_filter()
temp_filter = rs.temporal_filter()

try:
    while True:
        # Lấy khung hình từ camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Chuyển sang numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Làm mịn depth
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = spat_filter.process(depth_frame)
        depth_frame = temp_filter.process(depth_frame)
        depth_image = np.asanyarray(depth_frame.get_data())

        # Tiền xử lý ảnh màu
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Tìm contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Duyệt qua từng contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue  # Bỏ qua các vùng nhiễu nhỏ

            # Vẽ contour
            cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            roi_depth = depth_image[y:y+h, x:x+w]

            # Bỏ các điểm có giá trị 0 (không hợp lệ)
            valid_depth = roi_depth[roi_depth > 0]

            if valid_depth.size == 0:
                continue

            # Giả sử mặt bàn là vùng xung quanh contour
            border_size = 10
            table_area = depth_image[y+border_size:y+border_size+10, x+border_size:x+border_size+10]
            table_depth = table_area[table_area > 0]

            if table_depth.size == 0:
                continue

            avg_object_depth = np.median(valid_depth)
            avg_table_depth = np.median(table_depth)

            height_mm = avg_table_depth - avg_object_depth  # chênh lệch độ sâu

            cv2.putText(color_image, f"{height_mm:.1f}mm", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Hiển thị
        cv2.imshow("Color Image", color_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC để thoát
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
