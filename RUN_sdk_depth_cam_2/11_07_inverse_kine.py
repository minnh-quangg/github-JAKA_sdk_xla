import jkrc
import time
import numpy as np
import math
import math
from math import radians, degrees

TCP_Truc=1
UCS_Truc=9

def deg_to_rad(joint_deg):
    return [radians(x) for x in joint_deg]

def rad_to_deg(joint_rad):
    return [degrees(x) for x in joint_rad]

#Rx_cam,Ry_cam,Rz_cam ,speed=100, acc=20, tol=1
robot = jkrc.RC( "10.5.5.100")
ret = robot.login()     
if  ret == 0:  
    print('Trạng thái login: OK ')
    
robot.power_on()
robot.enable_robot()
# Giả sử robot đang ở vị trí khớp hiện tại:
current_joints = robot.get_joint_position()[1]
print(rad_to_deg(current_joints))

p_home=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
# P_EE_rxryrz = [161.517, 24.244, 127.994, -179.996, -0.003, -90.36]
P_EE_rxryrz = [161.235, 31.325, 187.996, math.radians(-179.996), math.radians(-0.003), math.radians(-90.36)]


status,joint_pose=robot.kine_inverse(p_home, P_EE_rxryrz)
print('trạng thái:',status)
print('trạng thái pose DEG:',(joint_pose))
print('trạng thái pose DEG:',rad_to_deg(joint_pose))


# import sys   
# import time        
# import jkrc  
  
# robot = jkrc.RC("192.168.31.15")#return a robot  
# ret = robot.login()#login   
# ret = robot.get_tcp_position()  
# if ret[0] == 0:  
#     print("the tcp position is :",ret[1])  
# else:  
#     print("some things happend,the errcode is: ",ret[0])  
# robot.logout()  #logout 
