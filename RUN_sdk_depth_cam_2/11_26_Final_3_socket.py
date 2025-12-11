import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import math
from math import radians
import time
import jkrc
import socket
import sys

# ==========================================
# CẤU HÌNH KẾT NỐI
# ==========================================
SERVER_IP = '192.168.1.100'  # IP Jaka 2
SERVER_PORT = 5000
ROBOT_IP = "192.168.31.15"      # IP Jaka 1

# ==========================================
# CẤU HÌNH ARUCO
# ==========================================
aruco_dict    = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters    = aruco.DetectorParameters()
detector      = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.02
n = 1

# --- CÁC HÀM HỖ TRỢ TÍNH TOÁN (GIỮ NGUYÊN) ---
def rotation_matrix_to_euler_xyz(R):
    if R[2, 0] < -1.0: R[2, 0] = -1.0
    if R[2, 0] > 1.0:  R[2, 0] = 1.0
    ry = math.asin(-R[2, 0])
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return rx, ry, rz

def median_depth_mm(depth_mm, cx, cy, k=5):
    h, w = depth_mm.shape[:2]
    r = k // 2
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r + 1, w), min(cy + r + 1, h)
    roi = depth_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0: return 0.0
    return float(np.median(valid))

class DepthSmoother:
    def __init__(self, alpha=0.3, max_jump=50):
        self.alpha, self.max_jump, self.last_val = alpha, max_jump, 0.0
    def update(self, new_val):
        if new_val <= 0: return self.last_val
        if self.last_val == 0: self.last_val = new_val
        elif abs(new_val - self.last_val) < self.max_jump:
            self.last_val = (1 - self.alpha) * self.last_val + self.alpha * new_val
        return self.last_val

def detect_marker_pose(timeout_s=10.0):
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy
    camera_matrix  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs    = np.array(color_intr.coeffs[:5], dtype=np.float32).reshape(-1, 1)
    smoother = DepthSmoother(alpha=0.25, max_jump=80)
    start_t  = time.time()
    pose_out = None
    marker_id_out = None
    try:
        while True:
            if (time.time() - start_t) > timeout_s:
                # Không in timeout để đỡ spam terminal khi chạy auto
                break
            frames  = pipe.wait_for_frames()
            aligned = rs.align(rs.stream.color).process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
                for i, (rvec, tvec, corner) in enumerate(zip(rvecs, tvecs, corners)):
                    cv2.drawFrameAxes(color, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                    pts = corner[0]
                    cx_pix, cy_pix = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    d_mm = smoother.update(median_depth_mm(depth, cx_pix, cy_pix))
                    if d_mm <= 0: continue
                    Z_m = d_mm / 1000.0
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(color_intr, [cx_pix, cy_pix], Z_m)
                    R, _ = cv2.Rodrigues(rvec)
                    rx, ry, rz = rotation_matrix_to_euler_xyz(R)
                    rx -= 3.14
                    if rx < -5.14: rx += 6.28
                    marker_id_out = int(ids[i][0])
                    pose_out = {'id': marker_id_out, 'x_mm': Xc*1000, 'y_mm': Yc*1000, 'z_mm': Zc*1000, 'rx_rad': rx, 'ry_rad': ry, 'rz_rad': rz}
                    break
            cv2.imshow("Jaka 1 Scanning...", color)
            if cv2.waitKey(1) & 0xFF == 27 or pose_out: break
    finally: pipe.stop(); cv2.destroyAllWindows()
    return pose_out, marker_id_out

def deg_to_rad(d): return [radians(x) for x in d]

def convert_camera_to_base_rotation(Rx, Ry, Rz):
    T = np.array([[1,0,0,125],[0,-1,0,98.5],[0,0,-1,280],[0,0,0,1]])
    def Rx_m(a): r=math.radians(a); return np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    def Ry_m(a): r=math.radians(a); return np.array([[math.cos(r),0,math.sin(r)],[0,1,0],[-math.sin(r),0,math.cos(r)]])
    def Rz_m(a): r=math.radians(a); return np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]])
    R_obj_cam = Rz_m(Rz) @ Ry_m(Ry) @ Rx_m(Rx)
    R_obj_base = T[:3,:3] @ R_obj_cam
    sy = max(-1.0, min(1.0, -R_obj_base[2,0]))
    return math.atan2(R_obj_base[2,1], R_obj_base[2,2]), math.asin(sy), math.atan2(R_obj_base[1,0], R_obj_base[0,0])

# =============================================================================
# HÀM ĐIỀU KHIỂN: GẮP - THẢ - CHỜ - CẤT
# =============================================================================
def run_full_cycle(robot, socket_conn, pose, ID):
    global n
    # --- 1. TÍNH TOÁN TỌA ĐỘ ---
    x, y, z_cam = pose['x_mm'], pose['y_mm'], pose['z_mm']
    rx, ry, rz = pose['rx_rad'], pose['ry_rad'], pose['rz_rad']
    
    print(f"📷 Marker ID: {ID}")
    
    p_cam = np.array([[x], [y], [z_cam], [1]])
    T = np.array([[1, 0, 0, 130], [0, -1, 0, 105], [0, 0, -1, 274], [0, 0, 0, 1]])
    P_EE_2 = (T @ p_cam)[:3,0] # Tọa độ Base
    
    # Góc xoay gắp vật
    r_UCS = [math.radians(-179.975), math.radians(1.307), math.radians(180.000)]
    rx_base, ry_base, rz_base = convert_camera_to_base_rotation(math.degrees(rx), math.degrees(ry), math.degrees(rz))
    
    # Các điểm cố định
    p_home = deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
    p_truocgap = [130, 85, 125]
    p_truocgap2 = [P_EE_2[0], P_EE_2[1], P_EE_2[2]+30]
    p_saugap = [P_EE_2[0], P_EE_2[1], P_EE_2[2]+100]
    p_truoctha = deg_to_rad([83.595, 59.935, 28.409, -0.777, 91.556, -49.196])

    # Chiều cao Z theo ID
    if ID == 3: z_val = 20
    elif ID == 2: z_val = 10
    elif ID == 4: z_val = 15
    else: z_val = 50

    # Điểm thả vật cho Jaka 2
    p_tam = [-266.058, 74.741, z_val]
    p_target_cartesian = np.hstack((p_tam, r_UCS))
    IO_cabinet = 0
    speed, acc, tol = 4, 1, 0.1

    # Tính IK
    _, joint_pose1 = robot.kine_inverse(p_home, np.hstack((p_truocgap, [rx_base, ry_base, rz_base])))
    _, joint_pose1_2 = robot.kine_inverse(p_home, np.hstack((p_truocgap2, [rx_base, ry_base, rz_base])))
    _, joint_pose2 = robot.kine_inverse(joint_pose1, np.hstack((P_EE_2, [rx_base, ry_base, rz_base]))) # Điểm gắp
    _, joint_pose3 = robot.kine_inverse(joint_pose2, np.hstack((p_saugap, [rx_base, ry_base, rz_base])))
    _, joint_target = robot.kine_inverse(joint_pose2, p_target_cartesian) # Điểm thả

    # ---------------------------------------------------
    # BƯỚC A: GẮP VẬT & ĐƯA SANG JAKA 2
    # ---------------------------------------------------
    print("🚀 A. Gắp vật và thả vào vùng vẽ...")
    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    
    # Đi gắp
    robot.joint_move_extend(joint_pos=joint_pose1, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=joint_pose1_2, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=joint_pose2, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    robot.set_digital_output(IO_cabinet, 0, 1) # Hút
    time.sleep(0.5)
    
    # Đi thả
    robot.joint_move_extend(joint_pos=joint_pose3, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=p_truoctha, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=joint_target, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    robot.set_digital_output(IO_cabinet, 0, 0) # Thả
    time.sleep(0.5)
    
    # Rút về Home để nhường chỗ
    robot.joint_move_extend(joint_pos=p_truoctha, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)

    # ---------------------------------------------------
    # BƯỚC B: GIAO TIẾP SOCKET
    # ---------------------------------------------------
    msg = f"{z_val}|1"
    socket_conn.send(msg.encode('utf-8'))
    print(f"📡 B. Đã gửi '{msg}' cho Jaka 2. Đang chờ vẽ...")

    # Vòng lặp chờ Done
    while True:
        try:
            data = socket_conn.recv(1024).decode('utf-8')
            if not data: 
                print("⚠️ Mất kết nối Jaka 2"); return
            print(f"📩 Nhận: {data}")
            if "Done" in data:
                print("⚡ Jaka 2 đã xong! Bắt đầu thu hồi.")
                break
        except: return

    # ---------------------------------------------------
    # BƯỚC C: THU HỒI VẬT & CẤT VÀO KHUÔN
    # ---------------------------------------------------
    print("📦 C. Thu hồi vật và cất vào khuôn...")
    
    # 1. Vào gắp lại vật (dùng joint_truoctha và joint_target đã tính ở trên)
    robot.joint_move_extend(joint_pos=p_truoctha, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol) # p_truoctha
    robot.joint_move_extend(joint_pos=joint_target, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)    # Chỗ vừa thả
    
    robot.set_digital_output(IO_cabinet, 0, 1) # Hút lại
    time.sleep(0.5)
    robot.joint_move_extend(joint_pos=p_truoctha, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol) # Nhấc lên

    # 2. Xác định vị trí khuôn (Logic cũ của bạn: z=10 -> khuôn 1, z=20 -> khuôn 2)
    if z_val == 10:
        # Khuôn 1
        p_truocluuvat = deg_to_rad([58.247, 29.582, 75.508, -0.243, 74.398, -164.102])
        p_luuvat = deg_to_rad([58.343, 34.727, 82.010, -0.264, 62.752, -163.949])
    elif z_val == 20 and n == 1:
        # Khuôn 2
        p_truocluuvat = deg_to_rad([-16.923, 49.488, 47.328, 1.798, 83.054, -149.908])
        p_luuvat = deg_to_rad([-16.851, 51.233, 52.182, 1.836, 76.455, -150.049])
        n = 2
    elif z_val == 20 and n == 2:
        # Mặc định (hoặc thêm logic khác)
        # p_truocluuvat = deg_to_rad([58.247, 29.582, 75.508, -0.243, 74.398, -164.102])
        # p_luuvat = deg_to_rad([58.343, 34.727, 82.010, -0.264, 62.752, -163.949])
        p_truocluuvat=deg_to_rad([-10.651, 61.548, 25.208, 1.762, 92.914, -143.326])
        p_luuvat=deg_to_rad([-10.579, 61.692, 33.049, 1.766, 84.930, -143.499])
        n = 1 

    # 3. Di chuyển vào khuôn
    robot.joint_move_extend(joint_pos=p_truocluuvat, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=p_luuvat, move_mode=0, is_block=True, speed=1, acc=0.25, tol=tol)
    
    robot.set_digital_output(IO_cabinet, 0, 0) # Thả vào khuôn
    time.sleep(0.5)
    
    # 4. Rút về & Về Home
    robot.joint_move_extend(joint_pos=p_truocluuvat, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    
    print("✅ HOÀN THÀNH CHU TRÌNH!")


# ==================== MAIN PROGRAM ==================== #
if __name__ == "__main__":
    # 1. Kết nối Robot
    robot = jkrc.RC(ROBOT_IP)
    if robot.login()[0] != 0: sys.exit("❌ Lỗi kết nối Robot")
    robot.power_on()
    robot.enable_robot()
    
    # 2. Kết nối Socket
    try:
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((SERVER_IP, SERVER_PORT))
        print(f"🔗 Đã kết nối Jaka 2 ({SERVER_IP})")
    except Exception as e:
        print(f"❌ Lỗi Socket: {e}"); sys.exit()

    print("\n>>> HỆ THỐNG TỰ ĐỘNG SẴN SÀNG <<<")
    
    # 3. Vòng lặp vô tận (Tự động)
    while True:
        # B1: Quét Marker (Timeout ngắn thôi để lặp nhanh nếu ko thấy)
        print("👀 Đang tìm marker...")
        pose, m_id = detect_marker_pose(timeout_s=5.0)
        
        if pose is None:
            # Không thấy marker -> Lặp lại tìm tiếp
            time.sleep(0.5)
            continue 
            
        # B2: Nếu thấy -> Chạy quy trình Full
        # (Pick -> Place -> Signal -> Wait -> Retrieve -> Store)
        run_full_cycle(robot, client_sock, pose, m_id)
        
        print("⏳ Nghỉ 2 giây trước khi tìm vật mới...")
        time.sleep(2)

    robot.logout()
    client_sock.close()