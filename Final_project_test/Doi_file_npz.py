import numpy as np

# Camera matrix và distortion coefficients mới
# camera_matrix = np.array( [[4.55399040e+03, 0.00000000e+00, 4.19539710e+02],
#  [0.00000000e+00, 3.59237838e+03, 1.79399700e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_matrix= np.array([[3.56996085e+03, 0.00000000e+00, 3.19112832e+02],
 [0.00000000e+00, 2.82240565e+03, 2.33814830e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],])
# dist_coeffs = np.array([[-1.76831236e+01, -7.30687750e+02,  3.43947836e-01, -4.70163261e-01,-5.66985201e+00]])
dist_coeffs = np.array([[-3.00745807e+00, -3.09703942e+03,  4.51261201e-02, -7.28036563e-02,-1.19835449e+01]])
# Lưu vào file npz
np.savez('camera_calib4.npz', camera_matrix=camera_matrix, dist_coeff=dist_coeffs)

print("Đã lưu camera matrix và distortion coefficients mới vào camera_calib2.npz")
