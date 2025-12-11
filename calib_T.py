import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pattern_size = (7, 7)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)             #Tạo ma trận zero để lưu tọa dộ 3D (x,y,z)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('data_images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    print(f"{fname} -> ret = {ret}")  
    
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        # cv.imshow('Detected Corners', img)
        cv.waitKey(50)
        
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

img = cv.imread('data_images/image_019.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow("Original", img)
cv.imshow("Undistorted", dst)
cv.waitKey(0)
cv.destroyAllWindows()