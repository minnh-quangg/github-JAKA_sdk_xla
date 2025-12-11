import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import math
from math import radians
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

########
def deg_to_rad(joint_deg):
    return [radians(x) for x in joint_deg]

def convert_camera_to_base_rotation(Rx_deg, Ry_deg, Rz_deg):
    # --- Hàm quay cơ bản ---
    # ==== Ví dụ dùng ====
    T = np.array([[1,  0,  0, 125],
              [0, -1,  0, 98.5],
              [0,  0, -1, 280],
              [0,  0,  0,   1]])
    def Rx(a):
        r = math.radians(a)
        return np.array([[1,0,0],
                         [0,math.cos(r), -math.sin(r)],
                         [0,math.sin(r),  math.cos(r)]])
    def Ry(a):
        r = math.radians(a)
        return np.array([[ math.cos(r), 0, math.sin(r)],
                         [0,1,0],
                         [-math.sin(r), 0, math.cos(r)]])
    def Rz(a):
        r = math.radians(a)
        return np.array([[math.cos(r), -math.sin(r), 0],
                         [math.sin(r),  math.cos(r), 0],
                         [0,0,1]])

    # --- Tạo ma trận quay của vật trong hệ camera (chuẩn ZYX) ---
    R_obj_cam = Rz(Rz_deg) @ Ry(Ry_deg) @ Rx(Rx_deg)

    # --- Lấy phần quay của ma trận T (camera → base) ---
    R_base_cam = T[:3, :3]

    # --- Tính quay của vật trong hệ base ---
    R_obj_base = R_base_cam @ R_obj_cam

    # --- Chuyển lại thành góc Euler theo cùng chuẩn ZYX ---
    sy = -R_obj_base[2,0]
    sy = max(-1.0, min(1.0, sy))
    Ry = (math.asin(sy))
    Rx = (math.atan2(R_obj_base[2,1], R_obj_base[2,2]))
    Rz = (math.atan2(R_obj_base[1,0], R_obj_base[0,0]))

    return Rx, Ry, Rz

# điều khiển robot
def pick_objects(P_EE_cam, Rotation_ojb2cam, speed=100, acc=20, tol=1):
    # thông số vị trí home
    home = [160, 25, 200, math.radians(180), math.radians(0), math.radians(-90)]
    p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
    # tinh toan chạy đến điểm vật trước
    p = np.array([[P_EE_cam[0]], [P_EE_cam[1]], [P_EE_cam[2]],[1]])
    T = np.array([[1, 0, 0, 125],
                  [0, -1, 0, 98.5],
                  [0, 0, -1, 280],
                  [0, 0, 0, 1]])
    P_EE_1= T @ p
    P_EE_2= P_EE_1[:3,0]
    P_EE_xyz = np.hstack((P_EE_2, [home[3], home[4], home[5]]))
    
    #-----#-------
    abs = 0
    incr = 1   


    #-----------------# 
    # robot = jkrc.RC( "192.168.31.15")
    robot = jkrc.RC( "10.5.5.100")
    ret = robot.login()     
    if  ret == 0:  
        print('Trạng thái login: OK ')
        
    robot.power_on()
    robot.enable_robot()
    
    # home = [160, 25, 200, math.radians(180), math.radians(0), math.radians(-90)]
    robot.linear_move(home, 0, False, speed)
    time.sleep(2)

    # move point
    ret, = robot.linear_move_extend(P_EE_xyz, 0, True, speed, acc, tol)
    
    rx_UCS,ry_UCS,rz_UCS=convert_camera_to_base_rotation(math.degrees(Rotation_ojb2cam[0]),math.degrees(Rotation_ojb2cam[1]),math.degrees(Rotation_ojb2cam[2]))
    P_EE_rxryrz = np.hstack((P_EE_2, [rx_UCS, ry_UCS, rz_UCS]))
    time.sleep(2)
    ret, = robot.linear_move_extend(P_EE_rxryrz, 0, True, speed, acc, tol)
    # print(ret)
    # time.sleep(100)
    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=0.2, acc=0.05, tol=0.1)
    # print("Robot đã hoàn thành di chuyển theo tất cả contour!")
    robot.logout()
    return ret


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
        
        P_cam = np.array([x_cam_mm, y_cam_mm, z_cam_mm])
        R_cam = np.array([rx_rad, ry_rad, rz_rad])
        
        confirm = input("Nhập 'y' để CHO ROBOT CHẠY đến pose này: ")
        if confirm.lower() == 'y':
            ret = pick_objects(P_cam, R_cam, speed=100, acc=20, tol=1)
            print("Robot return code =", ret)
            if ret == 0:
                print("✅ Trạng thái Robot: DONE")
        else:
            print("❌ Hủy, robot KHÔNG chạy.")



       