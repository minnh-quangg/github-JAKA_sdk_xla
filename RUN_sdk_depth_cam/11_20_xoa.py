import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import math
from math import radians
import time
import jkrc
def deg_to_rad(joint_deg):
    return [radians(x) for x in joint_deg]
def pick_objects(P_EE_cam, speed=100, acc=20, tol=1):
    # thông số vị trí home
    # home = [160, 25, 200, math.radians(180), math.radians(0), math.radians(-90)]
    p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
    # tinh toan chạy đến điểm vật trước
    p = np.array([[P_EE_cam[0]], [P_EE_cam[1]], [P_EE_cam[2]],[1]])
    T = np.array([[1, 0, 0, 130],
                  [0, -1, 0, 105],
                  [0, 0, -1, 280],
                  [0, 0, 0, 1]])
    P_EE_1= T @ p
    P_EE_2= P_EE_1[:3,0]
    print(P_EE_2)


    #-----------------# 
    # robot = jkrc.RC( "10.5.5.100")
    robot = jkrc.RC( "192.168.31.15")
    ret = robot.login()     
    if  ret == 0:  
        print('Trạng thái login: OK ')
        
    robot.power_on()
    robot.enable_robot()

    #----------------# HOME

    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=1, acc=0.25, tol=0.1)
    time.sleep(2)

    # move point
    P_EE_rxryrz = np.hstack((P_EE_2, [math.radians(-178), math.radians(0), math.radians(-90)]))
    # P_EE_rxryrz = [161.235, 31.325, 187.996, math.radians(-179.996), math.radians(-0.003), math.radians(-90.36)]

    status,joint_pose=robot.kine_inverse(p_home,P_EE_rxryrz)
    print('trạng thái:',status)
    

    robot.joint_move_extend(joint_pos=joint_pose, move_mode=0, is_block=True, speed=1, acc=0.25, tol=0.1)
    time.sleep(5)

    robot.joint_move_extend(joint_pos=p_home, move_mode=0, is_block=True, speed=1, acc=0.25, tol=0.1)
    # print("Robot đã hoàn thành di chuyển theo tất cả contour!")
    robot.logout()
    return ret


# ==================== MAIN ==================== #
P_EE_cam= [0, 50, 275]
pick_objects(P_EE_cam, speed=100, acc=20, tol=1)