import time
import jkrc

# loại IO (theo tài liệu)
IO_CABINET = 0   # control cabinet IO
IO_TOOL    = 1   # tool IO (tio)
IO_EXTEND  = 2   # extension / user integration IO

robot = jkrc.RC("ROBOT_IP")
robot.login()

# --- Nếu cần mở vout (nếu nguồn 24V là nguồn TIO v3) ---
# vout_enable: 0 = close, 1 = open
# vout_vol: 0 = 24V, 1 = 12V
robot.set_tio_vout_param(1, 0)   # bật vout, chọn 24V. (nếu robot hỗ trợ TIO v3)
time.sleep(0.1)

# --- Bật/chỉnh DO để nối Remote_ON với +24V ---
# Giả sử chân Remote_ON là  index 7 (ví dụ) hoặc chân tích hợp trên interface dùng IO_EXTEND.
# Trong tài liệu ví dụ, set_digital_output(IO_CABINET, 2, 1) set DO2 = 1. Ta dùng cấu trúc tương tự.
# Thay iotype và index theo sơ đồ chân/đánh số trên bảng phần cứng của bạn.
iotype = IO_EXTEND        # hoặc IO_TOOL / IO_CABINET tùy module chân nằm ở đâu
pin_index = 3             # <--- CHỈ LÀ VÍ DỤ: index = số chân (check numbering trên controller)
robot.set_digital_output(iotype, pin_index, 1)  # bật (1)
time.sleep(0.1)

# tắt
robot.set_digital_output(iotype, pin_index, 0)  # tắt (0)

robot.logout()
