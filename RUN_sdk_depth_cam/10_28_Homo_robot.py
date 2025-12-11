import cv2
import numpy as np
import os

# ==== Cấu hình người dùng ====
H_FILE     = "Homography_matrix.npz"   # file lưu ma trận H
CAM_INDEX  = 0                         # đổi camera
SCALE      = 2.0                       # hệ số phóng to hiển thị
WINDOW     = "Camera"                  # tên cửa sổ

# ==== Khai báo tọa độ thực tế tương ứng ====
# Đơn vị: cm - (theo thứ tự click)
REAL_POINTS_CM = np.array([
    [0, 25.2],
    [25.2, 20.1],
    [20.1, 0.0],
    [0.0,  0.0]
], dtype=np.float32)

# ==== Các hàm (function) ====
# Mở camera
def open_camera(index: int):
    """Mở camera"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera!")
    return cap

# Vẽ điểm
def draw_points(img, pts, scale: float):
    """Vẽ các điểm lên ảnh hiển thị"""
    for i, (x, y) in enumerate(pts):
        px, py = int(x * scale), int(y * scale)
        cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# Tính ma trận Homography
def compute_homography(img_points, real_points_cm):
    """Tính homography H từ 4 pixel <-> 4 điểm thực (cm)"""
    img_pts  = np.array(img_points, dtype=np.float32)
    real_pts = np.array(real_points_cm, dtype=np.float32)
    H, mask = cv2.findHomography(img_pts, real_pts, cv2.RANSAC)
    return H, mask

# Chuyển điểm pixel sang (X, Y) -> thực (cm)
def pixel_to_world(px, py, H):
    """Chuyển điểm pixel → (X, Y) thực (cm) bằng ma trận H"""
    pix = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pix, H)
    Xw, Yw = world[0][0]
    return float(Xw), float(Yw)

# ==== Khai báo Class ====
class HomographyApp:
    def __init__(self, h_file, cam_index, scale, real_pts_cm):
        self.h_file     = h_file
        self.cam_index  = cam_index
        self.scale      = scale
        self.real_pts   = real_pts_cm

        self.cap   = open_camera(self.cam_index)
        self.points = []  # lưu 4 pixel đã chọn
        self.H      = None

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._mouse_cb)

        print("[HƯỚNG DẪN]")
        print("- Click chuột trái chọn 4 điểm theo thứ tự.")
        print("- Đã có H: Click điểm bất kỳ đâu để xem (X,Y) thực (cm).")
        print("- Phím r: xóa H | s: lưu H | l: load H | q: thoát.")

    # Callback chuột
    def _mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x_real = int(x / self.scale)
        y_real = int(y / self.scale)

        if self.H is None:
            if len(self.points) < 4:
                self.points.append((x_real, y_real))
                print(f"Điểm {len(self.points)}: ({x_real}, {y_real})")
            if len(self.points) == 4:
                self.H, mask = compute_homography(self.points, self.real_pts)
                print("\n>>> Đã tính Homography (H):")
                print(self.H)
                print("Click thêm để xem tọa độ thực (cm):")
        else:
            Xw, Yw = pixel_to_world(x_real, y_real, self.H)
            print(f"Pixel ({x_real}, {y_real}) → Thực (X={Xw:.2f} cm, Y={Yw:.2f} cm)")

    def save_H(self):
        if self.H is None:
            print("Chưa có H để lưu!")
            return
        np.savez(self.h_file, H=self.H)
        print(f">>> Đã lưu H vào '{self.h_file}'")

    def load_H(self):
        if not os.path.exists(self.h_file):
            print(f"Không tìm thấy '{self.h_file}' để load.")
            return
        with np.load(self.h_file) as data:
            self.H = data["H"]
        print(">>> Đã load H từ file:")
        print(self.H)

    def reset_points(self):
        self.points = []
        self.H = None
        print(">>> Đã xóa 4 điểm và H. Hãy chọn lại 4 điểm.")

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("Không đọc được khung hình!")
                break

            disp = cv2.resize(frame, None, fx=self.scale, fy=self.scale)

            if self.H is None and self.points:
                draw_points(disp, self.points, self.scale)

            cv2.imshow(WINDOW, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_points()
            elif key == ord('s'):
                self.save_H()
            elif key == ord('l'):
                self.load_H()

        self.cap.release()
        cv2.destroyAllWindows()


# ==== Chạy chương trình ====
if __name__ == "__main__":
    app = HomographyApp(
        h_file=H_FILE,
        cam_index=CAM_INDEX,
        scale=SCALE,
        real_pts_cm=REAL_POINTS_CM
    )
    app.run()
