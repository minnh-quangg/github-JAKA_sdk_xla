import numpy as np

# Camera matrix và distortion coefficients mới
camera_matrix = np.array([[1.51570869e+03, 0.0, 2.87054983e+02],
                          [0.0, 1.21337949e+03, 2.37088764e+02],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([[-3.50852056e+00, 2.75336484e+02, -2.39808377e-02, -1.28219386e-02, -6.57833174e+03]])

# Lưu vào file npz
np.savez('camera_calib3.npz', camera_matrix=camera_matrix, dist_coeff=dist_coeffs)

print("Đã lưu camera matrix và distortion coefficients mới vào camera_calib2.npz")
