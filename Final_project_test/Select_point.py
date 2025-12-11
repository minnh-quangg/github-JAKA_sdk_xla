import cv2
import numpy as np

# ==== ĐỌC DỮ LIỆU CALIBRATION ====
with np.load('camera_calib4.npz') as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeff']

# ==== MỞ CAMERA ====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

# ==== LƯU CÁC ĐIỂM CLICK ====
points = []
done = False
scale = 1.5  # Hệ số phóng to cửa sổ hiển thị

def mouse_callback(event, x, y, flags, param):
    global points, done
    # Quy đổi tọa độ click từ ảnh hiển thị về ảnh gốc
    x_real = int(x / scale)
    y_real = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x_real, y_real))
            print(f"Điểm {len(points)}: ({x_real}, {y_real})")
        if len(points) == 4:
            done = True
            print(">>> Đã chọn đủ 4 điểm! Nhấn 's' để lưu hoặc ESC để thoát. <<<")

cv2.namedWindow("Select 4 Points (Undistorted)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Select 4 Points (Undistorted)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Select 4 Points (Undistorted)", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình!")
        break

    # --- Hiệu chỉnh méo ảnh ---
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # --- Phóng to ảnh để dễ click ---
    display = cv2.resize(undistorted, None, fx=scale, fy=scale)

    # --- Vẽ các điểm đã chọn ---
    for i, p in enumerate(points):
        px = int(p[0] * scale)
        py = int(p[1] * scale)
        cv2.circle(display, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(display, f"{i+1}", (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Select 4 Points (Undistorted)", display)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif done and key == ord('s'):
        print("\nTọa độ 4 điểm (sau calib):")
        for i, p in enumerate(points):
            print(f"Điểm {i+1}: {p}")
        np.savez('selected_points4.npz', points=np.array(points))
        print(">>> Đã lưu vào file selected_points.npz <<<")
        break

cap.release()
cv2.destroyAllWindows()
