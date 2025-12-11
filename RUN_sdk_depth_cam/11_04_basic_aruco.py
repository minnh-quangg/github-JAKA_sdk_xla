import cv2
import cv2.aruco as aruco
import numpy as np

camera_matrix = np.array([[614.13745, 0, 326.87924],
                          [0, 612.69507, 233.27437],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5,1))

cap = cv2.VideoCapture(1)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # aruco.drawDetectedMarkers(frame, corners, ids)
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            print(f"Marker pose: rvec={rvec.flatten()}, tvec={tvec.flatten()}")

    cv2.imshow("Webcam ArUco Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
