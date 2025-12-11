import cv2
import numpy as np

# ==== 1. Load dữ liệu calibration ====
with np.load('Data_camera_calib.npz') as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeff']

# ==== 2. Setup Camera ====
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

points = []      # lưu 4 điểm ảnh đã chọn
H = None         # ma trận homography
scale = 2.0      # hệ số phóng to cửa sổ hiển thị

# ==== 3. Hàm xử lý click chuột ====
def mouse_callback(event, x, y, flags, param):
    global points, H

    # quy đổi tọa độ click về ảnh gốc
    x_real, y_real = int(x/scale), int(y/scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        # nếu chưa có H thì đang ở bước chọn 4 điểm
        if H is None:
            if len(points) < 4:
                points.append((x_real, y_real))
                print(f"Điểm {len(points)}: ({x_real}, {y_real})")
            if len(points) == 4:
                # ==== Tính homography ====
                img_points = np.array(points, dtype=np.float32)

                # Tọa độ thực (cm) theo đúng thứ tự bạn chọn
                real_points = np.array([
                    [0,   14.4],
                    [18.3, 14.4],
                    [18.3, 0.0],
                    [0,   0.0]
                ], dtype=np.float32)

                H, mask = cv2.findHomography(img_points, real_points, cv2.RANSAC)
                np.savez('Homography_matrix.npz', H=H)
                print("\n>>> Đã chọn đủ 4 điểm, đã tính Homography và lưu Homography_matrix.npz")
                print(H)
                print("Click thêm điểm bất kỳ để xem tọa độ thực (cm).")

        else:
            # Nếu đã có H thì mỗi click → đổi pixel sang tọa độ thực
            pixel = np.array([[[x_real, y_real]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pixel, H)
            Xw, Yw = world[0][0]
            print(f"Pixel ({x_real}, {y_real}) → Thực tế (X={Xw:.2f} cm, Y={Yw:.2f} cm)")

# ==== 4. Cửa sổ hiển thị ====
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Camera", mouse_callback)

# ==== 5. Vòng lặp chính ====
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình!")
        break

    # Undistort
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Resize để click dễ hơn
    display = cv2.resize(undistorted, None, fx=scale, fy=scale)

    # Vẽ điểm đã chọn
    for i, p in enumerate(points):
        px, py = int(p[0]*scale), int(p[1]*scale)
        cv2.circle(display, (px, py), 8, (0,0,255), -1)
        cv2.putText(display, str(i+1), (px+10, py-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Camera", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
