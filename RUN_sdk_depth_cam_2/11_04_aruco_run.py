# import jkrc
# import numpy as np
# import math
# import time

# T = np.array([
#     [1,  0,  0, 127],
#     [0, -1,  0, 101],
#     [0,  0, -1, 280],
#     [0,  0,  0,   1]
# ])

# def pick_objects_from_manual_pose(x_cam_mm, y_cam_mm, z_cam_mm,
#                                   rx_deg, ry_deg, rz_deg,
#                                   speed=100, acc=20, tol=1):

#     p_cam = np.array([[x_cam_mm],
#                       [y_cam_mm],
#                       [z_cam_mm],
#                       [1]])

#     p_robot_h = T @ p_cam
#     p_robot   = p_robot_h[:3, 0] 

#     Rx_cam = math.radians(rx_deg)
#     Ry_cam = math.radians(ry_deg)
#     Rz_cam = math.radians(rz_deg)

#     P_EE = np.hstack((p_robot, [Rx_cam, Ry_cam, Rz_cam]))

#     print("\n=== Target in ROBOT frame (từ số bạn nhập) ===")
#     print(f"X={p_robot[0]:.1f} mm  Y={p_robot[1]:.1f} mm  Z={p_robot[2]:.1f} mm")
#     print(f"Rx={rx_deg:.2f}°  Ry={ry_deg:.2f}°  Rz={rz_deg:.2f}°")

#     # ===== 4. Điều khiển robot ===== #
#     robot = jkrc.RC("192.168.31.15")
#     ret = robot.login()
#     if ret == 0:
#         print("Login OK")
#     else:
#         print("Login FAIL, code =", ret)
#         return ret

#     robot.power_on()
#     robot.enable_robot()

#     home = [160, 25, 200,
#             math.radians(-180),
#             math.radians(0),
#             math.radians(-90)]

#     print("Go HOME...")
#     robot.linear_move(home, 0, False, speed)
#     time.sleep(1.0)

#     print("Go TARGET (manual pose)...")
#     ret, = robot.linear_move_extend(P_EE.tolist(), 0, False, speed, acc, tol)
#     print("Go target ret =", ret)

#     print("Back HOME...")
#     robot.linear_move_extend(home, 0, False, speed, acc, tol)

#     robot.logout()
#     return ret


# if __name__ == "__main__":
#     # ====== BƯỚC QUAN TRỌNG ======
#     # LẤY CÁC GIÁ TRỊ TỪ FILE aruco_get_pose.py VÀ NHẬP TAY Ở ĐÂY

#     # Ví dụ (bạn thay số lại theo kết quả thật):
#     x_cam_mm = 100.0   # <-- sửa theo X_cam (mm) detect được
#     y_cam_mm = 50.0    # <-- Y_cam (mm)
#     z_cam_mm = 300.0   # <-- Z_cam (mm)

#     rx_deg   = 0.0     # <-- Rx (deg) detect được
#     ry_deg   = 0.0     # <-- Ry (deg)
#     rz_deg   = -90.0   # <-- Rz (deg)

#     print("Số bạn đang dùng:")
#     print(f"X_cam={x_cam_mm} mm, Y_cam={y_cam_mm} mm, Z_cam={z_cam_mm} mm")
#     print(f"Rx={rx_deg}°, Ry={ry_deg}°, Rz={rz_deg}°")

#     confirm = input("Nhập 'y' để cho robot chạy với các giá trị này: ")
#     if confirm.lower() == 'y':
#         ret = pick_objects_from_manual_pose(
#             x_cam_mm, y_cam_mm, z_cam_mm,
#             rx_deg, ry_deg, rz_deg,
#             speed=100, acc=20, tol=1
#         )
#         print("Robot return code =", ret)
#     else:
#         print("❌ Hủy, robot KHÔNG chạy.")



import jkrc
import time
import numpy as np
import math

#Rx_cam,Ry_cam,Rz_cam ,speed=100, acc=20, tol=1
def pick_objects(P_EE_cam,Rotation_xyz ,speed=100, acc=20, tol=1):

    p = np.array([[P_EE_cam[0]], [P_EE_cam[1]], [P_EE_cam[2]],[1]])
    T = np.array([[1, 0, 0, 125],
                  [0, -1, 0, 98.5],
                  [0, 0, -1, 280],
                  [0, 0, 0, 1]])
    P_EE_1= T @ p
    P_EE_2= P_EE_1[:3,0]
    P_EE = np.hstack((P_EE_2, [Rotation_xyz[0], Rotation_xyz[1], Rotation_xyz[2]]))
    
    #-----#-------
    abs = 0
    incr = 1   
    robot = jkrc.RC( "192.168.31.15")
    ret = robot.login()     
    if  ret == 0:  
        print('Trạng thái login: OK ')
        
    robot.power_on()
    robot.enable_robot()
    
    home = [160, 25, 200, math.radians(180), math.radians(0), math.radians(-90)]
    robot.linear_move(home, 0, False, speed)
    time.sleep(2)

    # move point
    ret, = robot.linear_move_extend(P_EE, 0, False, speed, acc, tol)
    print(ret)
    # time.sleep(100)
    robot.linear_move_extend(home, 0, False, speed, acc, tol)
    # print("Robot đã hoàn thành di chuyển theo tất cả contour!")
    robot.logout()
    return ret


P=np.array([15.00, -42.98, 255])
R=np.array([math.radians(177.83),math.radians(2.39),math.radians(-0.96)])
ret=pick_objects(P, R, speed=100, acc=20, tol=1)
if ret == 0:
    print('Trạng thái Robot: DONE ')
