
import numpy as np
import cv2 as cv
import glob

# === 1. Tiêu chí dừng khi tìm góc chính xác ===
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# === 2. Chuẩn bị điểm góc trong không gian thực ===
# Bàn cờ 8x8 ô -> 7x7 giao điểm
pattern_size = (7, 7)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# === 3. Danh sách lưu điểm 3D và 2D ===
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

# === 4. Đọc tất cả ảnh calibration ===
images = glob.glob(r'D:\TT_totnghiep_JAKA\Python\calibimgs1\*.jpg')
# images = glob.glob(r'D:\TT_totnghiep_JAKA\captured_images\*.jpg')

#images = glob.glob('calibimgs1/*.jpg')


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # === 5. Tìm các góc bàn cờ ===
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    print(f"{fname} -> ret = {ret}")  
    
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        '''
        # Vẽ góc để kiểm tra
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('Detected Corners', img)
        cv.waitKey(500)
        '''
cv.destroyAllWindows() 


#CALIB
############################################################
# tìm mtx và dist
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print(f" RMS: {ret}")
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )
#chọn ảnh để calibration
real_gray = cv.imread(r'D:\TT_totnghiep_JAKA\Python\calibimgs1\image_005.jpg')
# real_resize =cv.resize(real_BGR, (640, 480))
# real_gray = cv.cvtColor(real_resize, cv.COLOR_BGR2GRAY)
h,  w = real_gray.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h),0, (w,h))
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha=0)

# xoá biến dạng
dst = cv.undistort(real_gray, mtx, dist, None, newcameramtx)
# x, y, w, h = roi
# real_calib = dst[y:y+h, x:x+w]
#real_calib =cv.resize(real_calib, (1280, 680))
cv.imshow("origin", real_gray)
# cv.imshow("calib", real_calib)
cv.imshow("calib", dst)
cv.waitKey(0)









# #########################################################

# #HOMOGRAPHY
# #########################################################
# # real_BGR = cv.imread(r'D:\TT_totnghiep_JAKA\Python\test5.jpg')
# # real_gray = cv.cvtColor(real_BGR, cv.COLOR_BGR2GRAY)
# standar_BGR =  cv.imread(r'D:\TT_totnghiep_JAKA\Python\banco2.jpg')
# standar_gray = cv.cvtColor(standar_BGR, cv.COLOR_BGR2GRAY)

# # Khởi tạo SIFT
# sift = cv.SIFT_create()

# # Phát hiện keypoints và descriptor
# keypoints1, descriptors1 = sift.detectAndCompute(standar_gray, None)
# keypoints2, descriptors2 = sift.detectAndCompute(real_calib, None)

# # Dùng BFMatcher để so khớp mô tả
# bf = cv.BFMatcher()
# matches = bf.knnMatch(descriptors1, descriptors2, k=2)
# #matches là 1 danh sách chứ các điêm [m,n] , m, n là 2 điểm keypoint gần nhất của des2 lên des1

# # sau đó ta đi lọc các matches tốt theo Lowe's ratio test
# good = []
# for m, n in matches:  # lọc qua từng cặp [m,n]
#     if m.distance < 0.75 * n.distance:
#         good.append(m)
# # --- Sau khi lọc good matches ---
# print(f"Số lượng match tốt: {len(good)}")

# # Hiển thị trực quan các cặp keypoint đã match
# match_vis = cv.drawMatches(
#     standar_BGR, keypoints1,        # ảnh chuẩn và keypoints
#     real_calib, keypoints2,           # ảnh thực và keypoints
#     good,                           # danh sách matches tốt
#     None,                           # output image (None = tự tạo)
#     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # chỉ vẽ match, không vẽ điểm đơn lẻ
# )

# cv.imshow("Keypoint Matches", match_vis)
# cv.waitKey(0)
# # Chuyển keypoint matching thành tọa độ
# if len(good) >= 4:  # phải lớn hơn hoặc bằng 4 là vì điều kiện của homography phải có 4 cặp điểm đặc trưng
#     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#     # Tìm homography 
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

#     # Vẽ khung trên ảnh scene
#     h, w = standar_gray.shape
#     pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#     scene_marked = real_calib.copy()
#     cv.polylines(scene_marked, [np.int32(dst)], True, (0, 255, 255), 3, cv.LINE_AA)
#     cv.imshow("Detected Chessboard", scene_marked)
#     cv.waitKey(0)
#     # Crop ảnh từ scene bằng phối cảnh
#     warped = cv.warpPerspective(real_calib, np.linalg.inv(M), (w, h))

#     # SHOW thẻ
#     cv.imshow("Trích xuất thẻ", warped)
#     cv.waitKey(0)


# #########################################################


# cv.destroyAllWindows()