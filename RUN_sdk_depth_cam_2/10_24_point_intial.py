import jkrc                 
import time 
import math

from math import radians

def deg_to_rad(joint_deg):
    return [radians(x) for x in joint_deg]

ABS = 0  
INCR= 1  

p=deg_to_rad([23.011, 3.551, 67.338, 0, 109.111, -203.260])

robot = jkrc.RC("192.168.31.15")
ret=robot.login()
print(ret)
robot.power_on() 
robot.enable_robot()  

print("Move Done")
robot.joint_move_extend(joint_pos=p, move_mode=0, is_block=True, speed=2, acc=2, tol=0.1)  
robot.logout() 

