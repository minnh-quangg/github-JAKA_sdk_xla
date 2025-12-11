import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
colorizer = rs.colorizer()  # tạo ảnh màu từ depth tự động

# lấy hệ số đổi đơn vị (z16 * depth_scale = mét)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

point = (320, 240)

def on_mouse(e, x, y, f, p):
    global point
    if e == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)

cv2.namedWindow("Depth Colormap")
cv2.setMouseCallback("Depth Colormap", on_mouse)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        # tạo ảnh colormap từ depth
        depth_color = np.asanyarray(colorizer.colorize(depth).get_data())
        bgr = np.asanyarray(color.get_data())

        # lấy khoảng cách tại pixel đang chọn (m -> mm)
        x, y = point
        if 0 <= x < depth.get_width() and 0 <= y < depth.get_height():
            raw = depth.get_distance(x, y)  # mét
            text = f"{raw*1000:.0f} mm" if raw > 0 else "No depth"
        else:
            text = "Out of frame"

        # vẽ lên cả 2 ảnh
        for img, name in [(depth_color, "Depth Colormap"), (bgr, "Color Frame")]:
            cv2.circle(img, point, 4, (0, 0, 255), -1)
            cv2.putText(img, text, (point[0]+8, point[1]-10),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 0), 2)
            cv2.imshow(name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("depth_colormap.png", depth_color)
            cv2.imwrite("color.png", bgr)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
