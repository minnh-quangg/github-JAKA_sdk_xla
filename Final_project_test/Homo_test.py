import cv2
import numpy as np

# === Đọc dữ liệu calib và homography ===
with np.load(r'D:\TT_totnghiep_JAKA\Python\Final_project\camera_calib4.npz') as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeff']

with np.load(r'D:\TT_totnghiep_JAKA\Python\Final_project\homography_matrix4.npz') as data:
    H = data['H']

# === Hàm callback chuột ===
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pixel, H)
        Xw, Yw = world[0][0]
        print(f"Pixel ({x:.1f}, {y:.1f})  →  Thực tế (X={Xw:.2f} cm, Y={Yw:.2f} cm)")

# === Mở camera ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Không mở được camera")
    exit()

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được khung hình!")
        break

    # Hiệu chỉnh méo
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Hiển thị ảnh
    cv2.imshow("Camera", undistorted)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
