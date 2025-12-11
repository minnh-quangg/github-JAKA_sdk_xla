# run_plane_rpy.py
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# ================== Tham số ==================
YOLO_WEIGHTS = "best.pt"     # đổi theo model của bạn
TARGET_CLASSES = None        # ví dụ ["square","circle"] hoặc để None -> lấy bbox lớn nhất
CONF_THRES = 0.5

# =============== Các hàm toán học ===============
def depth_to_points(depth_mm, intr):
    """Convert depth (mm) -> point cloud (m) trong hệ camera"""
    h, w = depth_mm.shape
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_mm.astype(np.float32) / 1000.0
    valid = z > 0
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    P = np.stack([x, y, z], axis=-1)  # (H,W,3)
    return P, valid

def fit_plane_pca(pts):
    c = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - c)
    n = Vt[-1]
    n = n / np.linalg.norm(n)
    if n[2] < 0:  # hướng normal ngửa về camera
        n = -n
    d = -np.dot(n, c)
    return n, d, c

def rotmat_from_normal(n, x_hint=np.array([1,0,0.], dtype=float)):
    z_axis = n / np.linalg.norm(n)
    x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
    if np.linalg.norm(x_proj) < 1e-6:
        x_hint = np.array([0,1,0.], dtype=float)
        x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
    x_axis = x_proj / np.linalg.norm(x_proj)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])  # R_cam_obj
    return R

def euler_from_R_XYZ(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        Rx = np.arctan2(R[2,1], R[2,2])
        Ry = np.arctan2(-R[2,0], sy)
        Rz = np.arctan2(R[1,0], R[0,0])
    else:
        Rx = np.arctan2(-R[1,2], R[1,1])
        Ry = np.arctan2(-R[2,0], sy)
        Rz = 0.0
    return Rx, Ry, Rz

def estimate_rpy_from_mask(depth_mm, intr, mask):
    P, valid = depth_to_points(depth_mm, intr)
    sel = (mask > 0) & valid
    pts = P[sel]
    if pts.shape[0] < 100:
        raise ValueError("Quá ít điểm để fit mặt phẳng")

    # lọc outlier thô
    n, d, _ = fit_plane_pca(pts)
    dist = np.abs(pts @ n + d)
    thr = np.quantile(dist, 0.8)  # giữ 80% điểm gần mặt phẳng
    pts_in = pts[dist < thr]

    n, d, _ = fit_plane_pca(pts_in)
    R = rotmat_from_normal(n)
    Rx, Ry, Rz = euler_from_R_XYZ(R)
    centroid = pts_in.mean(axis=0)
    return Rx, Ry, Rz, centroid, n

# =============== Camera + YOLO pipeline ===============
def main():
    # --- RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_stream = prof.get_stream(rs.stream.depth).as_video_stream_profile()
    intr = depth_stream.get_intrinsics()

    # --- YOLO ---
    model = YOLO(YOLO_WEIGHTS)

    cv2.namedWindow("View", cv2.WINDOW_NORMAL)
    print("[i] Nhấn Q để thoát.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_mm = np.asanyarray(depth_frame.get_data())  # uint16 (mm)

            # --- YOLO detect trên ảnh màu ---
            res = model.predict(source=color, conf=CONF_THRES, verbose=False)[0]

            # chọn bbox
            sel_bbox = None
            sel_cls = None
            if len(res.boxes) > 0:
                # lọc theo lớp nếu có
                candidates = []
                for b in res.boxes:
                    cls_name = model.names[int(b.cls)]
                    if (TARGET_CLASSES is None) or (cls_name in TARGET_CLASSES):
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        area = max(0, x2-x1) * max(0, y2-y1)
                        candidates.append((area, (x1,y1,x2,y2), cls_name))
                if candidates:
                    candidates.sort(reverse=True)  # lấy bbox lớn nhất
                    _, sel_bbox, sel_cls = candidates[0]

            overlay = color.copy()

            if sel_bbox is not None:
                x1,y1,x2,y2 = sel_bbox
                # đảm bảo nằm gọn trong ảnh
                h, w = depth_mm.shape
                x1 = np.clip(x1, 0, w-1); x2 = np.clip(x2, 0, w-1)
                y1 = np.clip(y1, 0, h-1); y2 = np.clip(y2, 0, h-1)

                # --- tạo mask từ bbox (có thể erode để giảm nền) ---
                mask = np.zeros_like(depth_mm, dtype=np.uint8)
                cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
                mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)

                try:
                    Rx, Ry, Rz, center, n = estimate_rpy_from_mask(depth_mm, intr, mask)
                    deg = np.degrees([Rx, Ry, Rz])

                    # vẽ thông tin
                    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
                    cx = int((x1+x2)/2); cy = int((y1+y2)/2)
                    cv2.circle(overlay, (cx,cy), 4, (0,0,255), -1)

                    text1 = f"Rx,Ry,Rz (deg): {deg[0]:.1f}, {deg[1]:.1f}, {deg[2]:.1f}"
                    text2 = f"centroid (m): x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}"
                    cv2.putText(overlay, text1, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(overlay, text2, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    # in ra console để bạn đọc sang robot
                    print(f"RPY_deg={deg.round(1)}  centroid_m={center.round(4)}  normal={n.round(3)}")

                except Exception as e:
                    cv2.putText(overlay, f"Loi fit mat phang: {e}", (10,25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("View", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
