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
    
    home = [160, 25, 200, math.radians(-180), math.radians(0), math.radians(-90)]
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


P_1=np.array([-60, -52, 244])
R=np.array([math.radians(-180),math.radians(0),math.radians(-90)])
ret=pick_objects(P_1,R,speed=100, acc=20, tol=1)
if ret == 0:
    print('Trạng thái Robot: DONE ')
