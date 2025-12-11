"""
Stable RealSense + ArUco pose detector (quaternion-based smoothing)
- Uses OpenCV ArUco detection but computes pose via cv2.solvePnP with
  cv2.SOLVEPNP_IPPE_SQUARE for improved stability on square markers.
- Filters depth (median + EMA) and orientation using quaternion smoothing
  (lerp in quaternion space + normalization).
- Outputs pose in camera frame: position (mm) and orientation both as
  quaternion and as Euler XYZ (radians).

How to use: run the script. It will open a RealSense stream and show a
preview window. Press ESC to quit or wait until a marker is detected.

Author: ChatGPT for JAKA robot (adapt to your use).
"""

import time
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs

# ---------- Parameters (tune as needed) ----------
MARKER_LENGTH_M = 0.05      # marker side length in meters (0.05 = 50 mm)
DEPTH_MED_K = 7             # median kernel for depth (odd)
DEPTH_ALPHA = 0.25          # EMA alpha for depth smoothing
DEPTH_MAX_JUMP = 80         # max jump in mm allowed for depth
QUAT_ALPHA = 0.20           # smoothing alpha for quaternion (0..1)
TIMEOUT_S = 15.0            # overall timeout waiting for a stable marker
CAM_W, CAM_H = 640, 480

# Use dictionary 4x4_50; change if using other marker family
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECT_PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECT_PARAMS)

# ---------- Utility: quaternion / rotation helpers ----------

def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    # Source: numerically stable conversion
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    return q / n


def quat_to_euler_xyz(q):
    # Convert quaternion (x,y,z,w) to Euler angles (rx, ry, rz) in radians, XYZ order
    x, y, z, w = q
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        ry = math.copysign(math.pi / 2, sinp)
    else:
        ry = math.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)
    return rx, ry, rz


def quat_smooth_lerp(q_old, q_new, alpha):
    # simple lerp in quaternion space then normalize (works well for small updates)
    if q_old is None:
        return quat_normalize(q_new)
    q = (1.0 - alpha) * q_old + alpha * q_new
    return quat_normalize(q)

# ---------- Depth utilities ----------

def median_depth_mm(depth_mm, cx, cy, k=5):
    h, w = depth_mm.shape[:2]
    r = k // 2
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r + 1, w), min(cy + r + 1, h)
    roi = depth_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))

class DepthSmoother:
    def __init__(self, alpha=DEPTH_ALPHA, max_jump=DEPTH_MAX_JUMP):
        self.alpha = alpha
        self.max_jump = max_jump
        self.last_val = 0.0

    def update(self, new_val):
        if new_val <= 0:
            return self.last_val
        if self.last_val == 0.0:
            self.last_val = new_val
        elif abs(new_val - self.last_val) < self.max_jump:
            self.last_val = (1.0 - self.alpha) * self.last_val + self.alpha * new_val
        return self.last_val

# ---------- Main detection function ----------

def detect_marker_pose(timeout_s=TIMEOUT_S):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    color_intr = color_stream.as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy
    camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array(color_intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)

    print("Camera matrix:\n", camera_matrix)
    print("Dist coeffs:\n", dist_coeffs.flatten())

    depth_smoother = DepthSmoother()
    quat_last = None
    start_t = time.time()
    pose_out = None

    # object points for a square marker in its own marker coordinate frame (z=0)
    s = MARKER_LENGTH_M / 2.0
    # Order: correspond to detected corners order from ArUco (corner[0] order)
    objp = np.array([[-s,  s, 0.0],
                     [ s,  s, 0.0],
                     [ s, -s, 0.0],
                     [-s, -s, 0.0]], dtype=np.float64)

    try:
        while True:
            if (time.time() - start_t) > timeout_s:
                print("⏱ Timeout, không thấy marker ổn định.")
                break

            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_mm = np.asanyarray(depth_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = DETECTOR.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(color, corners, ids)

                for i, c in enumerate(corners):
                    # c is shape (1,4,2)
                    pts = c[0].astype(np.float64)
                    # median depth at marker center
                    cx_pix = int(np.mean(pts[:, 0]))
                    cy_pix = int(np.mean(pts[:, 1]))
                    d_mm_raw = median_depth_mm(depth_mm, cx_pix, cy_pix, k=DEPTH_MED_K)
                    d_mm = depth_smoother.update(d_mm_raw)
                    if d_mm <= 0:
                        continue

                    # SolvePnP: use detected image points and known object points
                    imgp = pts.reshape(-1, 2)
                    # use IPPE_SQUARE for planar square markers (improves stability)
                    success, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not success:
                        # fallback to ITERATIVE
                        success, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    if not success:
                        continue

                    # draw axes for visualization (use 0.03m axis length)
                    cv2.drawFrameAxes(color, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_M * 0.6)

                    # produce rotation matrix and quaternion
                    R_mat, _ = cv2.Rodrigues(rvec)
                    q_new = rotmat_to_quat(R_mat)
                    quat_last = quat_smooth_lerp(quat_last, q_new, QUAT_ALPHA)

                    # compute position using deprojection (improve trust by using depth smoothed Z)
                    Z_m = d_mm / 1000.0
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(color_intr, [cx_pix, cy_pix], Z_m)
                    P_cam_mm = np.array([Xc * 1000.0, Yc * 1000.0, Zc * 1000.0])

                    rx, ry, rz = quat_to_euler_xyz(quat_last)
                    rx_deg, ry_deg, rz_deg = map(math.degrees, (rx, ry, rz))

                    marker_id = int(ids[i][0])

                    pose_out = {
                        'id': marker_id,
                        'x_mm': float(P_cam_mm[0]),
                        'y_mm': float(P_cam_mm[1]),
                        'z_mm': float(P_cam_mm[2]),
                        'quat_xyzw': quat_last.tolist(),
                        'rx_rad': float(rx),
                        'ry_rad': float(ry),
                        'rz_rad': float(rz)
                    }

                    # overlay text
                    txt = f"ID {marker_id} Z={d_mm:.0f}mm Rx={rx_deg:.1f} Ry={ry_deg:.1f} Rz={rz_deg:.1f}"
                    cv2.putText(color, txt, (cx_pix - 120, cy_pix - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # print console-friendly
                    print("\n===== STABLE POSE (camera frame) =====")
                    print(f"Marker ID       : {marker_id}")
                    print(f"X_cam (mm)      : {P_cam_mm[0]:.2f}")
                    print(f"Y_cam (mm)      : {P_cam_mm[1]:.2f}")
                    print(f"Z_cam (mm)      : {P_cam_mm[2]:.2f}")
                    print(f"Quat (x,y,z,w)  : {quat_last}")
                    print(f"Rx (deg)        : {rx_deg:.2f}")
                    print(f"Ry (deg)        : {ry_deg:.2f}")
                    print(f"Rz (deg)        : {rz_deg:.2f}")
                    print("================================")

                    # We break after first marker to return a single pose; remove if you want multi
                    break

            cv2.imshow('RealSense + ArUco Stable Pose', color)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or pose_out is not None:
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    return pose_out


if __name__ == '__main__':
    pose = detect_marker_pose(timeout_s=TIMEOUT_S)
    if pose is None:
        print('❌ Không tìm được marker ổn định trong thời gian cho phép.')
    else:
        print('\n✅ Đã detect xong, copy các giá trị để nhập vào code robot:')
        print(pose)
