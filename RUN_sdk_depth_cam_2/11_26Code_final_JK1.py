# =============================================================================
# FILE: jaka1_client.py
# ROBOT: JAKA 1 (Nhiệm vụ: Pick & Place & Store)
# VAI TRÒ SOCKET: Client (Kết nối tới Jaka 2)
# =============================================================================

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

# -----------------------------------------------------------------------------
# [CẤU HÌNH QUAN TRỌNG - BẠN CẦN KIỂM TRA]
# -----------------------------------------------------------------------------
SERVER_IP = '192.168.1.100'  # >>> IP CỦA CON JAKA 2 (SERVER) <<<
SERVER_PORT = 5000           # Port kết nối (Phải trùng với Jaka 2)
# ROBOT_IP_1 = "10.5.5.100"    # IP của con Jaka 1
ROBOT_IP_1 = "10.5.5.100"    # IP của con Jaka 1


# Marker setting
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.02 # Đơn vị mét

# -----------------------------------------------------------------------------
# PHẦN 1: CÁC HÀM HỖ TRỢ TÍNH TOÁN & CAMERA
# -----------------------------------------------------------------------------
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
    # Khởi tạo RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    
    # Lấy thông số nội tại Camera
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(color_intr.coeffs[:5], dtype=np.float32).reshape(-1, 1)
    
    align = rs.align(rs.stream.color)
    smoother = DepthSmoother(alpha=0.25, max_jump=80)
    start_t = time.time()
    
    pose_out = None
    marker_id_out = None
    
    try:
        while True:
            if (time.time() - start_t) > timeout_s:
                print("⏱ Timeout tìm marker."); break
                
            frames = pipe.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: continue
            
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                aruco.drawDetectedMarkers(color, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
                
                # Lấy marker đầu tiên tìm thấy
                rvec, tvec = rvecs[0], tvecs[0]
                marker_id_out = int(ids[0][0])
                
                # Tính tâm marker trên ảnh
                pts = corners[0][0]
                cx_pix = int(np.mean(pts[:, 0]))
                cy_pix = int(np.mean(pts[:, 1]))
                
                # Lấy depth tại tâm
                d_mm = smoother.update(median_depth_mm(depth, cx_pix, cy_pix))
                
                if d_mm > 0:
                    # Chuyển Pixel -> Camera 3D Coordinate
                    Z_m = d_mm / 1000.0
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(color_intr, [cx_pix, cy_pix], Z_m)
                    
                    # Tính Rotation
                    R, _ = cv2.Rodrigues(rvec)
                    rx, ry, rz = rotation_matrix_to_euler_xyz(R)
                    
                    # Xử lý góc (theo logic của bạn)
                    rx = rx - 3.14
                    if rx < -5.14: rx += 6.28
                    
                    pose_out = {
                        'x': Xc*1000, 'y': Yc*1000, 'z': Zc*1000, 
                        'rx': rx, 'ry': ry, 'rz': rz
                    }
                    
                    # Vẽ info lên ảnh
                    cv2.putText(color, f"ID:{marker_id_out} Z:{d_mm:.0f}", (cx_pix, cy_pix), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    
                    # Hiện ảnh 1 chút rồi break để trả về kết quả
                    cv2.imshow("Jaka 1 Eye", color)
                    cv2.waitKey(500)
                    break
            
            cv2.imshow("Jaka 1 Eye", color)
            if cv2.waitKey(1) & 0xFF == 27: break
            
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        
    return pose_out, marker_id_out

def deg_to_rad(d): return [radians(x) for x in d]

def convert_cam_to_base(Rx, Ry, Rz):
    # Ma trận chuyển đổi từ Camera -> Base Robot (Cần Calib chuẩn)
    T_cam_base = np.array([[1, 0, 0, 130], [0, -1, 0, 105], [0, 0, -1, 274], [0, 0, 0, 1]])
    
    def R_mat(a, axis):
        r = math.radians(a); c, s = math.cos(r), math.sin(r)
        if axis=='x': return np.array([[1,0,0],[0,c,-s],[0,s,c]])
        if axis=='y': return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        if axis=='z': return np.array([[c,-s,0],[s,c,0],[0,0,1]])
        
    R_obj_cam = R_mat(Rz, 'z') @ R_mat(Ry, 'y') @ R_mat(Rx, 'x')
    R_obj_base = T_cam_base[:3, :3] @ R_obj_cam
    
    sy = -R_obj_base[2,0]
    sy = max(-1.0, min(1.0, sy))
    
    return math.atan2(R_obj_base[2,1], R_obj_base[2,2]), math.asin(sy), math.atan2(R_obj_base[1,0], R_obj_base[0,0])

# -----------------------------------------------------------------------------
# PHẦN 2: CÁC HÀM ĐIỀU KHIỂN ROBOT (LOGIC CHÍNH)
# -----------------------------------------------------------------------------

# --- HÀM 1: Gắp vật & Đặt vào vùng vẽ ---
def robot_pick_and_place(robot, pose, m_id):
    print(f"🤖 [ROBOT] Gắp ID {m_id} -> Thả vào vùng vẽ")
    
    # 1. Tính toán tọa độ
    x, y, z = pose['x'], pose['y'], pose['z']
    rx, ry, rz = convert_cam_to_base(math.degrees(pose['rx']), math.degrees(pose['ry']), math.degrees(pose['rz']))
    
    # Chuyển tọa độ Cam -> Base
    p_cam = np.array([[x], [y], [z], [1]])
    T = np.array([[1, 0, 0, 130], [0, -1, 0, 105], [0, 0, -1, 274], [0, 0, 0, 1]])
    p_base = (T @ p_cam)[:3, 0]
    
    # Các điểm cố định
    p_home = deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
    
    # Điểm thả vật (Vùng vẽ của Jaka 2)
    z_place = 20 if m_id == 3 else (10 if m_id == 2 else (15 if m_id == 4 else 50))
    p_target_draw = [-266.058, 74.741, z_place, -3.14, 0.02, 3.14] # RxRyRz fix thẳng đứng
    
    # Điểm gắp
    p_pick = [p_base[0], p_base[1], p_base[2], rx, ry, rz]
    p_pick_up = [p_base[0], p_base[1], p_base[2]+100, rx, ry, rz]
    
    # --- THỰC HIỆN DI CHUYỂN ---
    # 1. Về Home
    robot.joint_move_extend(p_home, 0, True, 4, 1, 0.1)
    
    # 2. Đến gắp
    robot.linear_move_extend(p_pick_up, 0, False, 4, 1, 0.1)
    robot.linear_move_extend(p_pick, 0, True, 4, 1, 0.1)
    
    # Hút
    robot.set_digital_output(0, 1) 
    time.sleep(0.5)
    
    robot.linear_move_extend(p_pick_up, 0, False, 4, 1, 0.1) # Nhấc lên
    
    # 3. Đến thả
    p_target_up = [p_target_draw[0], p_target_draw[1], p_target_draw[2]+100, -3.14, 0.02, 3.14]
    robot.linear_move_extend(p_target_up, 0, False, 4, 1, 0.1)
    robot.linear_move_extend(p_target_draw, 0, True, 4, 1, 0.1)
    
    # Thả
    robot.set_digital_output(0, 0) 
    time.sleep(0.5)
    
    # 4. Rút về Home
    robot.linear_move_extend(p_target_up, 0, False, 4, 1, 0.1)
    robot.joint_move_extend(p_home, 0, True, 4, 1, 0.1)
    
    # Trả về vị trí thả để lát nữa gắp lại
    return p_target_draw, z_place

# --- HÀM 2: Cất vật vào khuôn (Sau khi vẽ xong) ---
def robot_store_object(robot, p_draw, z_val):
    print(f"🤖 [ROBOT] Lấy vật Z={z_val} từ vùng vẽ -> Cất vào khuôn")
    
    p_home = deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
    p_draw_up = [p_draw[0], p_draw[1], p_draw[2]+100, p_draw[3], p_draw[4], p_draw[5]]
    
    # 1. Đến gắp lại vật
    robot.linear_move_extend(p_draw_up, 0, False, 4, 1, 0.1)
    robot.linear_move_extend(p_draw, 0, True, 4, 1, 0.1)
    
    # Hút
    robot.set_digital_output(0, 1) 
    time.sleep(0.5)
    
    robot.linear_move_extend(p_draw_up, 0, False, 4, 1, 0.1)
    
    # 2. Đi đến khuôn (Dựa theo Z_ID)
    if z_val == 10:
        p_store = deg_to_rad([58.343, 34.727, 82.010, -0.264, 62.752, -163.949])
    else:
        p_store = deg_to_rad([49.418, 57.733, 39.910, -0.155, 81.815, -84.272])
        
    robot.joint_move_extend(p_store, 0, True, 4, 1, 0.1)
    
    # Thả
    robot.set_digital_output(0, 0) 
    time.sleep(0.5)
    
    # 3. Về Home
    robot.joint_move_extend(p_home, 0, True, 4, 1, 0.1)

# -----------------------------------------------------------------------------
# PHẦN 3: HÀM SOCKET (GỬI & NHẬN)
# -----------------------------------------------------------------------------
def send_signal(sock, z_height):
    """Gửi tín hiệu 'Z|1' cho Jaka 2"""
    msg = f"{z_height}|1"
    sock.send(msg.encode('utf-8'))
    print(f"📡 [SOCKET] Đã gửi: '{msg}' -> Chờ Jaka 2 vẽ...")

def wait_for_done(sock):
    """Chờ tín hiệu 'Done' từ Jaka 2"""
    print("⏳ [SOCKET] Đang chờ Jaka 2 báo 'Done'...")
    while True:
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data: return False
            
            print(f"📩 [SOCKET] Nhận: {data}")
            if "Done" in data:
                print("⚡ Jaka 2 đã xong! Bắt đầu thu hồi.")
                return True
        except: return False

# =============================================================================
# MAIN PROGRAM (JAKA 1)
# =============================================================================
if __name__ == "__main__":
    # 1. Kết nối Robot Jaka 1
    robot = jkrc.RC(ROBOT_IP_1)
    if robot.login() != 0: sys.exit("❌ Lỗi kết nối Robot Jaka 1")
    robot.power_on()
    robot.enable_robot()
    
    # 2. Kết nối Socket tới Jaka 2
    try:
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((SERVER_IP, SERVER_PORT))
        print(f"🔗 Đã kết nối Socket tới Jaka 2 ({SERVER_IP})")
    except Exception as e:
        print(f"❌ Lỗi kết nối Socket: {e}")
        sys.exit()

    # 3. Vòng lặp chính
    while True:
        print("\n========================================")
        print("   JAKA 1: SẴN SÀNG CHU TRÌNH MỚI")
        print("========================================")
        
        # B1: Tìm Marker
        pose, m_id = detect_marker_pose(timeout_s=30)
        if pose is None:
            print("⚠️ Không thấy marker. Thử lại..."); continue
            
        print(f"📷 Marker ID: {m_id}, Z: {pose['z']:.1f}mm")

        # B2: Gắp vật -> Thả vào chỗ vẽ -> Về Home
        # Hàm trả về vị trí thả (p_placed) để lát gắp lại
        p_placed, z_val = robot_pick_and_place(robot, pose, m_id)
        
        # B3: Gửi tín hiệu cho Jaka 2
        send_signal(client_sock, z_val)
        
        # B4: Chờ Jaka 2 làm xong
        if wait_for_done(client_sock):
            # B5: Jaka 2 xong -> Vào gắp lại -> Cất khuôn
            robot_store_object(robot, p_placed, z_val)
            print("✅ HOÀN THÀNH 1 CHU TRÌNH.")
        else:
            print("❌ Lỗi giao tiếp Socket. Dừng chương trình.")
            break
        
        time.sleep(2)

    robot.logout()
    client_sock.close()