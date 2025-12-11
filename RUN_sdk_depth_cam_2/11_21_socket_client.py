import socket
import threading
import sys
import time
import random # Chỉ dùng để giả lập số liệu, code thật bạn bỏ đi cũng được

# ================= CẤU HÌNH KẾT NỐI =================
HOST = '192.168.1.100'  # IP của Server (Ví dụ: Máy tính điều khiển trung tâm)
PORT = 5000

# ================= BIẾN TOÀN CỤC (GLOBAL) =================
# Đây là 2 biến quan trọng nhất bạn muốn gửi
object_height = 0.0  # Chiều cao vật (float)
task_status = 0      # 0: Chưa xong/Đang đo, 1: Hoàn thành

stop_event = threading.Event()

# ================= HÀM GỬI & NHẬN =================

def send_data(sock):
    """
    Hàm này lấy chiều cao và trạng thái hiện tại để gửi đi
    Format gửi: "chieucao|trangthai" (Ví dụ: "150.5|1")
    """
    global object_height, task_status
    
    try:
        # Đóng gói dữ liệu
        msg = f"{object_height}|{task_status}"
        sock.send(msg.encode('utf-8'))
        print(f"✅ [GỬI SERVER]: Chiều cao={object_height}mm | Trạng thái={task_status}")
    except Exception as e:
        print(f"❌ Lỗi gửi: {e}")

def receive_thread(sock):
    """Luồng nghe Server phản hồi"""
    while not stop_event.is_set():
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data:        



                print("\n⚠️ Mất kết nối Server.")
                stop_event.set()
                break
            print(f"\n📩 [SERVER PHẢN HỒI]: {data}")
            print("Nhập '1' để gửi kết quả đo, 'q' để thoát: ", end="", flush=True)
        except:
            break

# ================= LUỒNG XỬ LÝ CAMERA & ROBOT =================

def jaka_camera_process():
    """
    Đây là nơi bạn đặt code điều khiển Robot Jaka và Deep Cam.
    Nó sẽ chạy song song với việc gửi tin nhắn.
    """
    global object_height, task_status
    
    # >>> KHỞI TẠO CAMERA/ROBOT Ở ĐÂY <<<
    # Ví dụ: camera = DeepCam()
    # robot = JakaRobot()
    
    print("📷 Hệ thống Robot & Camera đang chạy...")

    while not stop_event.is_set():
        # -----------------------------------------------------------
        # >>> DÁN CODE XỬ LÝ ẢNH / ĐO CHIỀU CAO CỦA BẠN VÀO ĐÂY <<<
        # -----------------------------------------------------------
        
        # Giả sử đây là kết quả đo được từ hàm của bạn:
        # h = camera.get_depth_value() 
        
        # --- [MÔ PHỎNG] ---
        # Mình giả vờ đo được chiều cao thay đổi ngẫu nhiên từ 100mm đến 200mm
        simulated_h = random.uniform(20, 40)
        
        # CẬP NHẬT BIẾN TOÀN CỤC
        object_height = round(simulated_h, 2)
        
        # Logic trạng thái: Ví dụ nếu đo được chiều cao > 0 thì coi như xong (Status = 1)
        if object_height > 0:
            task_status = 1 
        else:
            task_status = 0
            
        time.sleep(0.1) # Nghỉ 1 chút để giảm tải CPU

# ================= CHƯƠNG TRÌNH CHÍNH =================

try:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    print(f"🤖 Đã kết nối tới Server điều khiển {HOST}:{PORT}")
except Exception as e:
    print(f"❌ Không thể kết nối Server: {e}")
    sys.exit()

# 1. Bật tai nghe (Nhận tin từ Server)
t_recv = threading.Thread(target=receive_thread, args=(client,), daemon=True)
t_recv.start()

# 2. Bật Robot & Camera (Đo đạc liên tục)
t_cam = threading.Thread(target=jaka_camera_process, daemon=True)
t_cam.start()

# 3. Vòng lặp chờ lệnh gửi của bạn
while not stop_event.is_set():
    try:
        # Bạn nhập 1 khi thấy Robot đã đo xong và muốn gửi báo cáo về Server
        check = input("Nhập '1' để gửi kết quả đo về Server, 'q' để thoát: ")
        
        if check == '1':
            if task_status == 1:
                send_data(client)
                # Tùy chọn: Sau khi gửi xong thì reset về 0 để đo vật mới?
                # task_status = 0 
            else:
                print("⚠️ Robot chưa hoàn thành đo (Status = 0). Vẫn muốn gửi? (y/n)")
                if input() == 'y': send_data(client)

        elif check.lower() == 'q':
            stop_event.set()
            client.close()
            break
    except:
        break