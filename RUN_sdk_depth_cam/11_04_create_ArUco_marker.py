import cv2
import cv2.aruco as aruco
import os

# Cấu hình
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

marker_size = 125            # kích thước phần mã (code area)
border_size = 20             # độ dày viền trắng (pixel)
save_folder = "Aruco_markers"
# save_folder = "Aruco_markers_6X6"

# Tạo folder nếu chưa có 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Tạo và lưu marker
for marker_id in range(10):  # tạo slg marker 
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # thêm viền trắng 
    marker_with_border = cv2.copyMakeBorder(
        marker_image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=255  
    )

    # lưu ảnh
    filename = os.path.join(save_folder, f"aruco_marker_{marker_id}.png")
    cv2.imwrite(filename, marker_with_border)
    print(f"Saved marker ID {marker_id} -> {filename}")

print("Done !")
