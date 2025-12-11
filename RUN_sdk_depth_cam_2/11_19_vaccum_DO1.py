# chưa sửa về vị trí chính xác
import jkrc                 
import time
import math

IO_cabinet = 0
IO_tool = 1
IO_extend = 2 

robot = jkrc.RC("10.5.5.100")
ret=robot.login()
print(ret)
robot.power_on() 
robot.enable_robot()  

ret = robot.get_digital_output(IO_cabinet, 0)
print(ret[1])
robot.set_digital_output(IO_cabinet, 0, 0) #Set pin output value DO1 to 1
time.sleep(0.1)
ret = robot.get_digital_output(IO_cabinet, 0)
print(ret[1])
robot.logout() 

