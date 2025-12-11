import jkrc
import time
import numpy as np
import math
import math
from math import radians, degrees

def deg_to_rad(joint_deg):
    return [radians(x) for x in joint_deg]

def rad_to_deg(joint_rad):
    return [degrees(x) for x in joint_rad]

robot = jkrc.RC( "10.5.5.100")
ret = robot.login()     
if  ret == 0:  
    print('Trạng thái login: OK ')
    
robot.power_on()
robot.enable_robot()
# Giả sử robot đang ở vị trí khớp hiện tại:
current_joints = robot.get_joint_position()[1]
print('Góc hiện tại:', rad_to_deg(current_joints))

p_home = deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
P_EE_rxryrz = [161.235, 31.325, 187.996, math.radians(-179.996), math.radians(-0.003), math.radians(-90.36)]

status, joint_pose=robot.kine_inverse(p_home, P_EE_rxryrz)
print('trạng thái:',status)
print('Pose Rad:',(joint_pose))
print('Pose Deg:',rad_to_deg(joint_pose))

