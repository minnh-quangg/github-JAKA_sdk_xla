import jkrc
import time
import numpy as np
import math

def pick_objects(P_EE_cam, Rotation_xyz, speed=100, acc=20, tol=1):
    p = np.array([[P_EE_cam[0]], [P_EE_cam[1]], [P_EE_cam[2]], [1]])
    T = np.array([[1, 0, 0, 125],
                  [0, -1, 0, 98.5],
                  [0, 0, -1, 280],
                  [0, 0, 0, 1]])
    P_EE_1 = T @ p
    P_EE_2 = P_EE_1[:3, 0]
    P_EE = np.hstack((P_EE_2, Rotation_xyz))
    
    robot = jkrc.RC("192.168.31.15")
    ret = robot.login()
    if ret == 0:
        print('Trạng thái login: OK')
    else:
        print('Login thất bại, code =', ret)
        return ret

    robot.power_on()
    robot.enable_robot()

    home = [160, 25, 200, math.radians(180), math.radians(0), math.radians(-90)]
    robot.linear_move(home, 0, False, speed)
    time.sleep(2)

    ret, = robot.linear_move_extend(P_EE, 0, False, speed, acc, tol)
    print("Kết quả di chuyển đến target:", ret)

    robot.linear_move_extend(home, 0, False, speed, acc, tol)
    robot.logout()
    return ret

# ==== Nhập từ bàn phím ====
try:
    x_cam_mm = float(input("Nhập X (mm): "))
    y_cam_mm = float(input("Nhập Y (mm): "))
    z_cam_mm = float(input("Nhập Z (mm): "))

    rx_deg = float(input("Nhập Rx (deg): "))
    ry_deg = float(input("Nhập Ry (deg): "))
    rz_deg = float(input("Nhập Rz (deg): "))

    # Chuyển độ sang radian
    R = np.array([math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)])
    P = np.array([x_cam_mm, y_cam_mm, z_cam_mm])

    # confirm = input("Nhập 'y' để cho robot chạy: ")
    # if confirm.lower() == 'y':
    ret = pick_objects(P, R, speed=100, acc=20, tol=1)
    if ret == 0:
        print("Trạng thái Robot: DONE ✅")
    else:
        print("❌ Hủy, robot KHÔNG chạy.")
except ValueError:
    print("Vui lòng nhập số hợp lệ!")
