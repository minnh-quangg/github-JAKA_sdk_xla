# chưa sửa về vị trí chính xác
import jkrc                 
import time
import math

from math import radians

def deg_to_rad(joint_deg):
    return [radians(x) for x in joint_deg]

ABS = 0  
INCR= 1  

p=deg_to_rad([15.041, 16.657, 60.623, -0.226, 101.920, -207.417])
# p=deg_to_rad([83.65, 59.416, 29.712, -0.745, 90.807, -136.095])

# robot = jkrc.RC("192.168.31.15")
robot = jkrc.RC("10.5.5.100")

ret=robot.login()
print(ret)
robot.power_on() 
robot.enable_robot()  

print("Move Done")
robot.joint_move_extend(joint_pos=p, move_mode=0, is_block=True, speed=4, acc=1, tol=0.1)  
robot.logout() 

