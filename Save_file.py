import cv2
import os

# === 1. Tạo folder lưu ảnh nếu chưa có ===
save_folder = "data_images"
os.makedirs(save_folder, exist_ok=True)

# === 2. Khởi tạo webcam (0 = camera mặc định) ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

counter = 0  # biến đếm số ảnh

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận frame, thoát...")
        break

    frame_resized = cv2.resize(frame, (640, 480))
    # gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Camera (Nhan SPACE de chup, ESC de thoat)", gray)

    cv2.imshow("Camera (Nhan SPACE de chup, ESC de thoat)", frame_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC -> thoát
        break
    elif key == 32:  # SPACE -> chụp ảnh
        filename = f"image_{counter:03d}.jpg"   # tạo tên dạng image_000, image_001
        filepath = os.path.join(save_folder, filename)
        # cv2.imwrite(filepath,gray)
        cv2.imwrite(filepath,frame_resized)
        print(f"Đã lưu: {filepath}")
        counter += 1  # tăng biến đếm

cap.release()
cv2.destroyAllWindows()
