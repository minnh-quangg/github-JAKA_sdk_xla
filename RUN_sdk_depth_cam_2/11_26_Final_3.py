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

marker_length = 0.02
z = None
joint_target = None
joint_truoctha = None
n=1


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
    # --- LẤY DEPTH SCALE TỪ SENSOR ---
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()   # mét / raw_unit
    # print("Depth scale (m/unit):", depth_scale)

    align   = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    color_intr   = color_stream.as_video_stream_profile().get_intrinsics()

    fx, fy, cx, cy = color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy
    camera_matrix  = np.array([[fx, 0,   cx],
                               [0,  fy,  cy],
                               [0,  0,   1 ]], dtype=np.float32)
    dist_coeffs    = np.array(color_intr.coeffs[:5],
                              dtype=np.float32).reshape(-1, 1)

    smoother = DepthSmoother(alpha=0.25, max_jump=80)
    start_t  = time.time()
    pose_out = None

    # --- BIẾN ỔN ĐỊNH 5S ---
    stable_start_time = None
    required_stable_duration = 3.0 # Thời gian cần ổn định (giây)

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
                # --- LOGIC ĐẾM THỜI GIAN ---
                if stable_start_time is None:
                    stable_start_time = time.time()
                
                elapsed = time.time() - stable_start_time
                remaining = required_stable_duration - elapsed
                
                # Vẽ thông báo lên chính frame hiện tại (không tạo frame mới)
                color_text = (0, 255, 255) # Vàng
                status_txt = f"Hold Stable: {remaining:.1f}s"
                
                if remaining <= 0:
                    status_txt = "CAPTURED!"
                    color_text = (0, 255, 0) # Xanh lá

                cv2.putText(color, status_txt, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_text, 2)
                # ---------------------------

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
                    
                    # Nếu chưa đủ 5 giây, chỉ update filter, không lấy pose và không break
                    if remaining > 0:
                        continue

                    # --- KHI ĐÃ ĐỦ 5 GIÂY ---
                    Z_m = d_mm_filt / 1000.0
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(
                        color_intr, [cx_pix, cy_pix], Z_m
                    )
                    P_cam = np.array([Xc * 1000.0,
                                      Yc * 1000.0,
                                      Zc * 1000.0])  # mm

                    R, _       = cv2.Rodrigues(rvec)
                    rx, ry, rz = rotation_matrix_to_euler_xyz(R)
                    # ### Trực
                    rx=(rx-3.14)
                    if rx < -2*2.57:
                        rx = rx+2*3.14
                    
                    # ###
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
                    
                    # Đã lấy được pose sau 5s, thoát vòng for và while
                    break
                
                # Thoát vòng lặp while nếu đã có pose_out
                if pose_out is not None:
                    break
            
            else:
                # Mất marker -> Reset thời gian
                stable_start_time = None
                cv2.putText(color, "No Marker Found", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # CHỈ DÙNG 1 FRAME DUY NHẤT
            cv2.imshow("RealSense + ArUco Pose", color)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC hoặc đã có pose
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    
    # Xử lý trả về
    mk_id_ret = pose_out['id'] if pose_out is not None else -1
    return pose_out, mk_id_ret

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
def pick_objects(P_EE_cam, Rotation_ojb2cam, speed, acc, tol,delay, ID):
    
    # tinh toan chạy đến điểm vật trước
    p = np.array([[P_EE_cam[0]], [P_EE_cam[1]], [P_EE_cam[2]],[1]])
    T = np.array([[1, 0, 0, 130],
                  [0, -1, 0, 105],
                  [0, 0, -1, 274],
                  [0, 0, 0, 1]])
    P_EE_1= T @ p
    P_EE_2= P_EE_1[:3,0]
    r_UCS = [math.radians(-179.975),math.radians(1.307),math.radians(180.000)]
    rx_UCS,ry_UCS,rz_UCS=convert_camera_to_base_rotation(math.degrees(Rotation_ojb2cam[0]),math.degrees(Rotation_ojb2cam[1]),math.degrees(Rotation_ojb2cam[2]))
    
    # khởi tạo p
    p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
    p_truocgap= [130, 85, 125]
    p_truocgap2= [P_EE_2[0],P_EE_2[1],P_EE_2[2]+30]
    p_saugap= [P_EE_2[0],P_EE_2[1],P_EE_2[2]+100]

    p_truoctha=deg_to_rad([83.595, 59.935, 28.409, -0.777, 91.556, -49.196])
    if ID == 3:
        z = 20
    elif ID == 2:
        z = 10
    elif ID == 4:
        z = 15
    else:
        z = 50


          
    

    p_tam = [-266.058, 74.741 , z]
    p_target = np.hstack((p_tam, r_UCS))
    # p_target=deg_to_rad([56.676, 56.463, 44.457, -0.517, 78.424, -185.076])
    # p_sautha=deg_to_rad([])
    
    #-----#-------y
    abs = 0
    incr = 1 
    IO_cabinet = 0
    IO_tool = 1
    IO_extend = 2   


    #-----------------# 
    # robot = jkrc.RC("192.168.31.15")
    robot = jkrc.RC("10.5.5.100")
    ret = robot.login()     
    if  ret == 0:  
        print('Trạng thái login: OK ')
        
    robot.power_on()
    robot.enable_robot()

    #----------------# HOME
    # P_EE_rxryrz = [161.235, 31.325, 187.996, math.radians(-179.996), math.radians(-0.003), math.radians(-90.36)]

    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    time.sleep(2)

    # ------- #tính toán chuyển động
    # tính toán điểm chuyển động trung gian trước khi gắp vậy

    P_truocgap_rxryrz = np.hstack((p_truocgap, [rx_UCS, ry_UCS, rz_UCS]))
    status1,joint_pose1=robot.kine_inverse(p_home,P_truocgap_rxryrz)
    P_truocgap2_rxryrz = np.hstack((p_truocgap2, [rx_UCS, ry_UCS, rz_UCS]))
    status1,joint_pose1_2=robot.kine_inverse(p_home,P_truocgap2_rxryrz)
    # print('trạng thái:',status1)

    # tính toán điểm gắp vật 
    
    P_EE_rxryrz = np.hstack((P_EE_2, [rx_UCS, ry_UCS, rz_UCS]))
    status2,joint_pose2=robot.kine_inverse(joint_pose1,P_EE_rxryrz)
    # print('trạng thái:',status2)
    
    # tính toán điểm chuyển động trung gian sau khi gắp vật

    P_saugap_rxryrz = np.hstack((p_saugap, [rx_UCS, ry_UCS, rz_UCS]))
    status3,joint_pose3=robot.kine_inverse(joint_pose2,P_saugap_rxryrz)
    # tính toán điểm thả vật

   
    status4,joint_target=robot.kine_inverse(joint_pose2,p_target)
    # print('trạng thái:',status1)
    
    # MOVE GẤP
      # trước gấp
    robot.joint_move_extend(joint_pos=joint_pose1, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    # time.sleep(delay)
    robot.joint_move_extend(joint_pos=joint_pose1_2, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
      # gấp vật
    robot.joint_move_extend(joint_pos=joint_pose2, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    time.sleep(delay)
      # Hút
    robot.set_digital_output(IO_cabinet, 0, 1)
    time.sleep(delay)
      # sau gấp
    robot.joint_move_extend(joint_pos=joint_pose3, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    # time.sleep(delay)
      # trước thả
    robot.joint_move_extend(joint_pos=p_truoctha, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    # time.sleep(delay)
      # vị trí thả
    robot.joint_move_extend(joint_pos=joint_target, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
      # tắt hút
    robot.set_digital_output(IO_cabinet, 0, 0)
    time.sleep(delay)
      # sau thả
    robot.joint_move_extend(joint_pos=p_truoctha, move_mode=0, is_block=False, speed=speed, acc=acc, tol=tol)
    
    
    # time.sleep(delay)

    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=speed, acc=acc, tol=tol)
    # print("Robot đã hoàn thành di chuyển theo tất cả contour!")
    robot.logout()
    return ret , joint_target,p_truoctha,z, P_EE_2[2]


# ==================== MAIN ==================== #
if __name__ == "__main__":
    
    while True:    
        tam = int(input("biến trạng thái: "))
        print("biến trạng thái là:", tam)
        if tam==1:
            print(">>> ĐÃ VÀO CHẾ ĐỘ 1")
            # 1) Lấy pose từ ArUco + RealSense
            pose,marker_ID = detect_marker_pose(timeout_s=40.0)

            if pose is None:
                print("❌ Không tìm được marker nào, robot KHÔNG chạy.")
            else:
                x_cam_mm = pose['x_mm']
                y_cam_mm = pose['y_mm']
                z_cam_mm = pose['z_mm']
                rx_rad   = pose['rx_rad']
                ry_rad   = pose['ry_rad']
                rz_rad   = pose['rz_rad']
                
                print(math.degrees(rx_rad),math.degrees(ry_rad),math.degrees(rz_rad))
                P_cam = np.array([x_cam_mm, y_cam_mm, z_cam_mm])
                R_cam = np.array([rx_rad, ry_rad, rz_rad])
                ret, joint_target, joint_truoctha , z, z_return = pick_objects(P_cam, R_cam, speed=4, acc=1, tol=0.1,delay=0.5,ID=marker_ID)
                time.sleep(1)
                tam=0
        elif tam == 2:
            print(">>> ĐÃ VÀO CHẾ ĐỘ 2")
            if  z == 10:
                p_truocluuvat1=deg_to_rad([58.247, 29.582, 75.508, -0.243, 74.398, -164.102])
                p_luuvat1=deg_to_rad([58.343, 34.727, 82.010, -0.264, 62.752, -163.949])
                p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])

                # robot = jkrc.RC("192.168.31.15")
                IO_cabinet = 0
                robot = jkrc.RC("10.5.5.100")
                ret = robot.login()     
                robot.power_on()
                robot.enable_robot()
                # trước thả
                robot.joint_move_extend(joint_pos=joint_truoctha, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                # time.sleep(delay)
                # vị trí thả
                robot.joint_move_extend(joint_pos=joint_target, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                robot.set_digital_output(IO_cabinet, 0, 1)
                time.sleep(1)
                robot.joint_move_extend(joint_pos=joint_truoctha, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_truocluuvat1, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_luuvat1, move_mode=0, is_block=True, speed=1, acc=0.25, tol=0.1)
                robot.set_digital_output(IO_cabinet, 0, 0)
                time.sleep(0.5)
                robot.joint_move_extend(joint_pos=p_truocluuvat1, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                time.sleep(2)
                pose,marker_ID = detect_marker_pose(timeout_s=40.0)

                if pose is None:
                    print("❌ Không tìm được marker nào, robot KHÔNG chạy.")
                else:
                    x_cam_mm = pose['x_mm']
                    y_cam_mm = pose['y_mm']
                    z_cam_mm = pose['z_mm']
                    rx_rad   = pose['rx_rad']
                    ry_rad   = pose['ry_rad']
                    rz_rad   = pose['rz_rad']
                    
                    print(math.degrees(rx_rad),math.degrees(ry_rad),math.degrees(rz_rad))
                    P_cam = np.array([x_cam_mm, y_cam_mm, z_cam_mm])
                    R_cam = np.array([rx_rad, ry_rad, rz_rad])
                    ret, joint_target, joint_truoctha , z, z_return = pick_objects(P_cam, R_cam, speed=4, acc=1, tol=0.1,delay=0.5,ID=marker_ID)
                    time.sleep(1)

                robot.logout()
                
                




            elif z ==20  and n==1:
                p_truocluuvat2=deg_to_rad([-10.651, 61.548, 25.208, 1.762, 92.914, -143.326])
                p_luuvat2=deg_to_rad([-10.579, 61.692, 33.049, 1.766, 84.930, -143.499])
                p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])

                # robot = jkrc.RC("192.168.31.15")
                IO_cabinet = 0
                robot = jkrc.RC("10.5.5.100")
                ret = robot.login()     
                robot.power_on()
                robot.enable_robot()
                # trước thả
                robot.joint_move_extend(joint_pos=joint_truoctha, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                # time.sleep(delay)
                # vị trí thả
                robot.joint_move_extend(joint_pos=joint_target, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                robot.set_digital_output(IO_cabinet, 0, 1)
                time.sleep(1)
                robot.joint_move_extend(joint_pos=joint_truoctha, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_truocluuvat2, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_luuvat2, move_mode=0, is_block=True, speed=1, acc=0.25, tol=0.1)
                robot.set_digital_output(IO_cabinet, 0, 0)
                time.sleep(0.5)
                robot.joint_move_extend(joint_pos=p_truocluuvat2, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                time.sleep(2)
                pose,marker_ID = detect_marker_pose(timeout_s=40.0)

                if pose is None:
                    print("❌ Không tìm được marker nào, robot KHÔNG chạy.")
                else:
                    x_cam_mm = pose['x_mm']
                    y_cam_mm = pose['y_mm']
                    z_cam_mm = pose['z_mm']
                    rx_rad   = pose['rx_rad']
                    ry_rad   = pose['ry_rad']
                    rz_rad   = pose['rz_rad']
                    
                    print(math.degrees(rx_rad),math.degrees(ry_rad),math.degrees(rz_rad))
                    P_cam = np.array([x_cam_mm, y_cam_mm, z_cam_mm])
                    R_cam = np.array([rx_rad, ry_rad, rz_rad])
                    ret, joint_target, joint_truoctha , z, z_return = pick_objects(P_cam, R_cam, speed=4, acc=1, tol=0.1,delay=0.5,ID=marker_ID)
                    time.sleep(1)
                robot.logout()
                tam=0
                n=2
                print('n:',n)
            elif z ==20  and n == 2:
                p_truocluuvat3=deg_to_rad([-16.923, 49.488, 47.328, 1.798, 83.054, -149.908])
                p_luuvat3=deg_to_rad([-16.851, 51.233, 52.182, 1.836, 76.455, -150.049])
                p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])

                # robot = jkrc.RC("192.168.31.15")
                IO_cabinet = 0
                robot = jkrc.RC("10.5.5.100")
                ret = robot.login()     
                robot.power_on()
                robot.enable_robot()
                # trước thả
                robot.joint_move_extend(joint_pos=joint_truoctha, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                # time.sleep(delay)
                # vị trí thả
                robot.joint_move_extend(joint_pos=joint_target, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                robot.set_digital_output(IO_cabinet, 0, 1)
                time.sleep(1)
                robot.joint_move_extend(joint_pos=joint_truoctha, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_truocluuvat3, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_luuvat3, move_mode=0, is_block=True, speed=1, acc=0.25, tol=0.1)
                robot.set_digital_output(IO_cabinet, 0, 0)
                time.sleep(0.5)
                robot.joint_move_extend(joint_pos=p_truocluuvat3, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)
                robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=False, speed=4, acc=1, tol=0.1)
                time.sleep(2)
                pose,marker_ID = detect_marker_pose(timeout_s=40.0)

                if pose is None:
                    print("❌ Không tìm được marker nào, robot KHÔNG chạy.")
                else:
                    x_cam_mm = pose['x_mm']
                    y_cam_mm = pose['y_mm']
                    z_cam_mm = pose['z_mm']
                    rx_rad   = pose['rx_rad']
                    ry_rad   = pose['ry_rad']
                    rz_rad   = pose['rz_rad']
                    
                    print(math.degrees(rx_rad),math.degrees(ry_rad),math.degrees(rz_rad))
                    P_cam = np.array([x_cam_mm, y_cam_mm, z_cam_mm])
                    R_cam = np.array([rx_rad, ry_rad, rz_rad])
                    ret, joint_target, joint_truoctha , z, z_return = pick_objects(P_cam, R_cam, speed=4, acc=1, tol=0.1,delay=0.5,ID=marker_ID)
                    time.sleep(1)
                robot.logout()
                tam=0
                n=1
            else:
                tam=0