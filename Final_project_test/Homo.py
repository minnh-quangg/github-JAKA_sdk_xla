import numpy as np
import cv2

# === Đọc điểm ảnh (pixel) đã chọn ===
data = np.load(r'D:\TT_totnghiep_JAKA\Python\Final_project\selected_points4.npz')
img_points = data['points'].astype(np.float32)

# === Khai báo tọa độ thực tế tương ứng (theo thứ tự bạn click) ===
# Đơn vị: cm
real_points = np.array([
    [0, 14.4],
    [18.3, 14.4],
    [18.3, 0.0],
    [0, 0.0]
], dtype=np.float32)

# === Tính ma trận homography ===
H, mask = cv2.findHomography(img_points, real_points, cv2.RANSAC)

print("Ma trận Homography (H):")
print(H)

# === Lưu lại để sử dụng sau ===
np.savez(r'D:\TT_totnghiep_JAKA\Python\Final_project\homography_matrix4.npz', H=H)
print("\n>>> Đã lưu H vào file 'homography_matrix1.npz' <<<")

# --- Ví dụ: kiểm tra chuyển đổi ---
test_pixel = np.array([[img_points[0]]], dtype=np.float32)
mapped = cv2.perspectiveTransform(test_pixel, H)
print(f"\nĐiểm pixel {img_points[0]} tương ứng tọa độ thật {mapped[0][0]}")
