import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import math
import time

aruco_dict    = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters    = aruco.DetectorParameters()
detector      = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.05  

def rotation_matrix_to_euler_xyz(R):
    if R[2, 0] < -1.0:
        R[2, 0] = -1.0
    if R[2, 0] > 1.0:
        R[2, 0] = 1.0

    ry = math.asin(-R[2, 0])
    cy = math.cos(ry)

    if abs(cy) > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0

    return rx, ry, rz  # radian


# ==================== MEDIAN DEPTH TRONG K×K ==================== #
def median_depth_mm(depth_mm, cx, cy, k=5):
    h, w = depth_mm.shape[:2]
    r = k // 2
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r + 1, w), min(cy + r + 1, h)
    roi = depth_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


# ==================== LỌC THEO THỜI GIAN (EMA) ==================== #
class DepthSmoother:
    def __init__(self, alpha=0.3, max_jump=50):
        self.alpha = alpha
        self.max_jump = max_jump
        self.last_val = 0.0

    def update(self, new_val):
        if new_val <= 0:
            return self.last_val
        if self.last_val == 0:
            self.last_val = new_val
        elif abs(new_val - self.last_val) < self.max_jump:
            self.last_val = (1 - self.alpha) * self.last_val + self.alpha * new_val
        return self.last_val


def detect_marker_pose(timeout_s=30.0, stability_duration=5.0):
    """
    timeout_s: Thời gian tối đa cho phép chạy chương trình nếu không tìm thấy gì.
    stability_duration: Thời gian cần giữ marker liên tục để chốt giá trị (giây).
    """
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align   = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    color_intr   = color_stream.as_video_stream_profile().get_intrinsics()

    fx, fy, cx, cy = color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy
    camera_matrix  = np.array([[fx, 0,   cx],
                               [0,  fy,  cy],
                               [0,  0,   1 ]], dtype=np.float32)
    dist_coeffs    = np.array(color_intr.coeffs[:5],
                              dtype=np.float32).reshape(-1, 1)

    print("Camera matrix:\n", camera_matrix)
    print("Dist coeffs:\n", dist_coeffs.flatten())
    print(f"👉 Vui lòng giữ marker ổn định trong {stability_duration} giây để lấy dữ liệu...")

    smoother = DepthSmoother(alpha=0.25, max_jump=80)
    start_t  = time.time()
    pose_out = None
    
    # Biến theo dõi thời gian ổn định
    stable_start_time = None 

    try:
        while True:
            # Kiểm tra timeout tổng (để tránh treo chương trình mãi mãi)
            if (time.time() - start_t) > timeout_s:
                print("⏱ Timeout tổng, không hoàn thành task.")
                break

            frames  = pipe.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color    = np.asanyarray(color_frame.get_data())
            depth_mm = np.asanyarray(depth_frame.get_data())
            gray     = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                # --- LOGIC ĐẾM THỜI GIAN ỔN ĐỊNH ---
                if stable_start_time is None:
                    stable_start_time = time.time() # Bắt đầu đếm
                
                elapsed_stable = time.time() - stable_start_time
                remaining_time = stability_duration - elapsed_stable
                
                # Vẽ thanh progress hoặc text đếm ngược
                if remaining_time > 0:
                    msg = f"Hold still: {remaining_time:.1f}s"
                    color_txt = (0, 255, 255) # Màu vàng khi đang chờ
                else:
                    msg = "CAPTURED!"
                    color_txt = (0, 255, 0)   # Màu xanh khi xong
                
                cv2.putText(color, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, color_txt, 2)
                # -----------------------------------

                aruco.drawDetectedMarkers(color, corners, ids)
                rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                # Lấy marker đầu tiên tìm thấy để xử lý
                rvec, tvec, corner = rvecs[0], tvecs[0], corners[0]
                
                cv2.drawFrameAxes(color, camera_matrix, dist_coeffs,
                                    rvec, tvec, 0.03)

                pts = corner[0]
                cx_pix = int(np.mean(pts[:, 0]))
                cy_pix = int(np.mean(pts[:, 1]))

                d_mm_raw  = median_depth_mm(depth_mm, cx_pix, cy_pix, k=7)
                d_mm_filt = smoother.update(d_mm_raw)

                if d_mm_filt > 0:
                    Z_m = d_mm_filt / 1000.0
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                        color_intr, [cx_pix, cy_pix], Z_m
                    )
                    P_cam = np.array([Xc * 1000.0, Yc * 1000.0, Zc * 1000.0])  # mm

                    R, _       = cv2.Rodrigues(rvec)
                    rx, ry, rz = rotation_matrix_to_euler_xyz(R)
                    
                    # ### Trực transformation
                    rx = (rx - 3.14)
                    if rx < -2 * 2.57:
                        rx = rx + 2 * 3.14
                    
                    marker_id = int(ids[0][0])
                    rx_deg, ry_deg, rz_deg = map(math.degrees, [rx, ry, rz])

                    # Hiển thị thông số realtime lên màn hình
                    txt_pose = (f"ID:{marker_id} Z:{d_mm_filt:.0f}mm "
                                f"Rx:{rx_deg:.0f} Ry:{ry_deg:.0f} Rz:{rz_deg:.0f}")
                    cv2.putText(color, txt_pose, (cx_pix - 80, cy_pix - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # --- KIỂM TRA ĐIỀU KIỆN HOÀN THÀNH ---
                    if elapsed_stable >= stability_duration:
                        # Đã đủ 5 giây, lưu giá trị cuối cùng
                        pose_out = {
                            'id'    : marker_id,
                            'x_mm'  : P_cam[0],
                            'y_mm'  : P_cam[1],
                            'z_mm'  : P_cam[2],
                            'rx_rad': rx,
                            'ry_rad': ry,
                            'rz_rad': rz
                        }
                        
                        print("\n===== FINAL STABLE POSE =====")
                        print(f"Marker ID       : {marker_id}")
                        print(f"X_cam (mm)      : {P_cam[0]:.2f}")
                        print(f"Y_cam (mm)      : {P_cam[1]:.2f}")
                        print(f"Z_cam (mm)      : {P_cam[2]:.2f}")
                        print(f"Rx (deg)        : {rx_deg:.2f}")
                        print(f"Ry (deg)        : {ry_deg:.2f}")
                        print(f"Rz (deg)        : {rz_deg:.2f}")
                        print("=============================")
                        
                        # Break khỏi vòng lặp while
                        break 

            else:
                # Không thấy marker -> Reset bộ đếm thời gian
                stable_start_time = None
                cv2.putText(color, "No Marker", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("RealSense + ArUco (Hold 5s)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC để thoát cưỡng bức
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    return pose_out


if __name__ == "__main__":
    # Đặt timeout tổng lớn hơn một chút (30s) để bạn có thời gian chỉnh marker
    pose = detect_marker_pose(timeout_s=60.0, stability_duration=5.0)
    
    if pose is None:
        print("❌ Không hoàn thành hoặc không tìm thấy marker.")
    else:
        print("\n✅ Đã lấy được dữ liệu sau 5s ổn định.")