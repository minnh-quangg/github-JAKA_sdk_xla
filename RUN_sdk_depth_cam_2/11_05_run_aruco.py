import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import math
import time
import jkrc

aruco_dict    = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters    = aruco.DetectorParameters()
detector      = aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.05 


# R -> euler xyz
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


# lọc median
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


# lọc smooth
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


def detect_marker_pose(timeout_s=10.0):
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

    smoother = DepthSmoother(alpha=0.25, max_jump=80)
    start_t  = time.time()
    pose_out = None

    try:
        while True:
            if (time.time() - start_t) > timeout_s:
                print("⏱ Timeout, không thấy marker nào ổn định.")
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
                aruco.drawDetectedMarkers(color, corners, ids)
                rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                for i, (rvec, tvec, corner) in enumerate(zip(rvecs, tvecs, corners)):
                    cv2.drawFrameAxes(color, camera_matrix, dist_coeffs,
                                      rvec, tvec, 0.03)

                    pts = corner[0]
                    cx_pix = int(np.mean(pts[:, 0]))
                    cy_pix = int(np.mean(pts[:, 1]))

                    d_mm_raw  = median_depth_mm(depth_mm, cx_pix, cy_pix, k=7)
                    d_mm_filt = smoother.update(d_mm_raw)

                    if d_mm_filt <= 0:
                        continue

                    Z_m = d_mm_filt / 1000.0
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                        color_intr, [cx_pix, cy_pix], Z_m
                    )
                    P_cam = np.array([Xc * 1000.0,
                                      Yc * 1000.0,
                                      Zc * 1000.0])  # mm

                    R, _       = cv2.Rodrigues(rvec)
                    rx, ry, rz = rotation_matrix_to_euler_xyz(R)

                    marker_id = int(ids[i][0])

                    pose_out = {
                        'id'    : marker_id,
                        'x_mm'  : P_cam[0],
                        'y_mm'  : P_cam[1],
                        'z_mm'  : P_cam[2],
                        'rx_rad': rx,
                        'ry_rad': ry,
                        'rz_rad': rz
                    }

                    rx_deg, ry_deg, rz_deg = map(math.degrees, [rx, ry, rz])

                    txt = (f"ID {marker_id} "
                           f"Z={d_mm_filt:.0f}mm "
                           f"Rx={rx_deg:.1f} "
                           f"Ry={ry_deg:.1f} "
                           f"Rz={rz_deg:.1f}")
                    cv2.putText(color, txt, (cx_pix - 80, cy_pix - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    # In ra console cho bạn copy
                    print("\n===== POSE (camera frame) =====")
                    print(f"Marker ID       : {marker_id}")
                    print(f"X_cam (mm)      : {P_cam[0]:.2f}")
                    print(f"Y_cam (mm)      : {P_cam[1]:.2f}")
                    print(f"Z_cam (mm)      : {P_cam[2]:.2f}")
                    print(f"Rx (deg)        : {rx_deg:.2f}")
                    print(f"Ry (deg)        : {ry_deg:.2f}")
                    print(f"Rz (deg)        : {rz_deg:.2f}")
                    print("================================")

                    # lấy 1 marker là đủ để chạy robot
                    break

            cv2.imshow("RealSense + ArUco Pose (ONLY DETECT)", color)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or pose_out is not None:  # ESC hoặc đã có pose
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    return pose_out


# điều khiển robot
def pick_objects(P_EE_cam, Rotation_xyz, speed=100, acc=20, tol=1):
    """
    P_EE_cam: [x_cam_mm, y_cam_mm, z_cam_mm] trong hệ camera (mm)
    Rotation_xyz: [Rx_rad, Ry_rad, Rz_rad] trong hệ camera (radian)
    """

    # 1) Đổi điểm từ camera frame -> robot base frame
    p = np.array([[P_EE_cam[0]],
                  [P_EE_cam[1]],
                  [P_EE_cam[2]],
                  [1]])

    T = np.array([[1,  0,  0, 125],
                  [0, -1,  0, 98.5],
                  [0,  0, -1, 280],
                  [0,  0,  0,   1]])

    P_EE_1 = T @ p
    P_EE_2 = P_EE_1[:3, 0]
    print()

    # 2) Ghép pose cuối cùng (base frame + góc rad)
    P_EE = np.hstack((P_EE_2,
                      [Rotation_xyz[0], Rotation_xyz[1], Rotation_xyz[2]]))

    print("\n=== Target in ROBOT frame (từ ArUco) ===")
    print(f"X={P_EE_2[0]:.1f} mm  Y={P_EE_2[1]:.1f} mm  Z={P_EE_2[2]:.1f} mm")
    print(f"Rx={math.degrees(Rotation_xyz[0]):.2f}°  "
          f"Ry={math.degrees(Rotation_xyz[1]):.2f}°  "
          f"Rz={math.degrees(Rotation_xyz[2]):.2f}°")

    # 3) Kết nối robot
    robot = jkrc.RC("10.5.5.100")
    login_ret = robot.login()
    print("login_ret =", login_ret)

    # login_ret có thể là 0 hoặc (0,)
    if isinstance(login_ret, (tuple, list)):
        login_code = login_ret[0]
    else:
        login_code = login_ret

    if login_code != 0:
        print("❌ Login FAIL, code =", login_code)
        return login_code
    print("✅ Trạng thái login: OK")

    robot.power_on()
    robot.enable_robot()

    home = [160, 25, 200,
            math.radians(180),
            math.radians(0),
            math.radians(-90)]

    print("Go HOME...")
    robot.linear_move(home, 0, False, speed)
    time.sleep(2)

    print("Go TARGET (ArUco pose)...")
    move_ret = robot.linear_move_extend(P_EE.tolist(), 0, False, speed, acc, tol)
    print("raw move_ret =", move_ret)

    if isinstance(move_ret, (tuple, list)):
        move_code = move_ret[0]
    else:
        move_code = move_ret

    print("Go target ret =", move_code)

    print("Back HOME...")
    robot.linear_move_extend(home, 0, False, speed, acc, tol)

    robot.logout()
    return move_code


# ==================== MAIN ==================== #
if __name__ == "__main__":
    # 1) Lấy pose từ ArUco + RealSense
    pose = detect_marker_pose(timeout_s=10.0)

    if pose is None:
        print("❌ Không tìm được marker nào, robot KHÔNG chạy.")
    else:
        x_cam_mm = pose['x_mm']
        y_cam_mm = pose['y_mm']
        z_cam_mm = pose['z_mm']
        rx_rad   = pose['rx_rad']
        ry_rad   = pose['ry_rad']
        rz_rad   = pose['rz_rad']

        print("\n=== Pose dùng để chạy robot (camera frame) ===")
        print(f"X_cam={x_cam_mm:.2f} mm, Y_cam={y_cam_mm:.2f} mm, Z_cam={z_cam_mm:.2f} mm")
        print(f"Rx={math.degrees(rx_rad):.2f}°, "
              f"Ry={math.degrees(ry_rad):.2f}°, "
              f"Rz={math.degrees(rz_rad):.2f}°")
        print(x_cam_mm,y_cam_mm,z_cam_mm)
        P_cam = np.array([x_cam_mm, y_cam_mm, z_cam_mm])
        R_cam = np.array([rx_rad, ry_rad, rz_rad])
        print(x_cam_mm,y_cam_mm,z_cam_mm)
        confirm = input("Nhập 'y' để CHO ROBOT CHẠY đến pose này: ")
        if confirm.lower() == 'y':
            ret = pick_objects(P_cam, R_cam, speed=100, acc=20, tol=1)
            print("Robot return code =", ret)
            if ret == 0:
                print("✅ Trạng thái Robot: DONE")
        else:
            print("❌ Hủy, robot KHÔNG chạy.")
