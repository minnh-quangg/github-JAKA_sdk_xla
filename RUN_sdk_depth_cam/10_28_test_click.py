import pyrealsense2 as rs
import numpy as np
import cv2

# 1) Khởi tạo pipeline + align depth -> color
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# 2) Lấy intrinsics của DEPTH
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
depth_intr = depth_stream.get_intrinsics()

click_pt = [None]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt[0] = (x, y)

cv2.namedWindow("Color")
cv2.setMouseCallback("Color", on_mouse)

MM_PER_M = 1000.0  # hệ số đổi mét -> mm

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())

        if click_pt[0] is not None:
            u, v = click_pt[0]

            # Z_m là mét; đổi sang mm để deproject hay giữ mét đều được
            Z_m = depth_frame.get_distance(u, v)  # meters
            if Z_m > 0:
                # Deproject dùng đơn vị mét
                X_m, Y_m, Z_m = rs.rs2_deproject_pixel_to_point(depth_intr, [u, v], Z_m)

                # Đổi toàn bộ sang mm
                X_mm = X_m * MM_PER_M
                Y_mm = Y_m * MM_PER_M
                Z_mm = Z_m * MM_PER_M

                print(f"Click ({u},{v}) -> XYZ (mm): X={X_mm:.1f}, Y={Y_mm:.1f}, Z={Z_mm:.1f}")

                cv2.circle(color_img, (u, v), 5, (0, 255, 0), -1)
                cv2.putText(color_img,
                            f"X:{X_mm:.1f} Y:{Y_mm:.1f} Z:{Z_mm:.1f} mm",
                            (u+8, v-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Color", color_img)
        if cv2.waitKey(1) == 27:  # ESC
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
