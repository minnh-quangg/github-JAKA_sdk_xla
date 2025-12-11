import jkrc
import cv2, time, math

import numpy as np
import pyrealsense2 as rs

from ultralytics import YOLO
from collections import deque
from realsense_camera import RealsenseCamera

PRIORITY = ["Heart", "circle", "square"]   # thứ tự: trái tim → tròn → vuông

# ==== Median depth kxk around center point ==== #
def median_depth_mm(depth_mm, cx, cy, k=5):
    h, w = depth_mm.shape[:2]
    r = k // 2
    x1 = max(cx - r, 0); y1 = max(cy - r, 0)
    x2 = min(cx + r + 1, w); y2 = min(cy + r + 1, h)
    roi = depth_mm[y1:y2, x1:x2]
    if roi.size == 0: return 0.0
    valid = roi[roi > 0]
    if valid.size == 0: return 0.0
    return float(np.median(valid))

# ==== Each new object assign -> unique track ==== #
class Track:
    __slots__ = ("tid","cx","cy","age","miss","depth_hist")
    def __init__(self, tid, cx, cy):
        self.tid = tid
        self.cx, self.cy = cx, cy
        self.age = 0
        self.miss = 0
        self.depth_hist = deque(maxlen=20)
    def update(self, cx, cy, d_mm):
        self.cx, self.cy = cx, cy
        self.age += 1
        self.miss = 0
        if d_mm > 0:
            self.depth_hist.append(d_mm)
    def predict_miss(self):
        self.miss += 1
    def avg_mm(self):
        return float(np.mean(self.depth_hist)) if len(self.depth_hist)>0 else 0.0

# ==== Ổn định frame, đo độ sâu trung bình ==== #
class CentroidTracker:
    def __init__(self, max_distance=60, max_age=30):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
    @staticmethod
    def _dist(a,b):
        ax,ay=a; bx,by=b
        dx=ax-bx; dy=ay-by
        return (dx*dx+dy*dy)**0.5
    def update(self, detections):
        det_centers = [(d['cx'], d['cy']) for d in detections]
        tids = list(self.tracks.keys())
        tr_centers = [(self.tracks[tid].cx, self.tracks[tid].cy) for tid in tids]
        assigned_tracks = [None]*len(detections)

        if not tr_centers:
            for i,d in enumerate(detections):
                t = Track(self.next_id, d['cx'], d['cy'])
                t.update(d['cx'], d['cy'], d['d_mm'])
                self.tracks[self.next_id] = t
                assigned_tracks[i] = t
                self.next_id += 1
            return assigned_tracks

        D = np.zeros((len(tr_centers), len(det_centers)), dtype=np.float32)
        for i,tc in enumerate(tr_centers):
            for j,dc in enumerate(det_centers):
                D[i,j] = self._dist(tc,dc)

        used_tracks=set(); used_dets=set()
        for _ in range(min(len(tr_centers), len(det_centers))):
            min_val=1e9; min_i=-1; min_j=-1
            for i in range(len(tr_centers)):
                if i in used_tracks: continue
                for j in range(len(det_centers)):
                    if j in used_dets: continue
                    if D[i,j] < min_val:
                        min_val, min_i, min_j = D[i,j], i, j
            if min_i==-1: break
            if min_val <= self.max_distance:
                tid = tids[min_i]
                t = self.tracks[tid]
                d = detections[min_j]
                t.update(d['cx'], d['cy'], d['d_mm'])
                assigned_tracks[min_j] = t
                used_tracks.add(min_i); used_dets.add(min_j)
            else:
                break

        for j in range(len(detections)):
            if j in used_dets: continue
            d = detections[j]
            t = Track(self.next_id, d['cx'], d['cy'])
            t.update(d['cx'], d['cy'], d['d_mm'])
            self.tracks[self.next_id] = t
            assigned_tracks[j] = t
            self.next_id += 1

        for i,tid in enumerate(tids):
            if i not in used_tracks:
                self.tracks[tid].predict_miss()

        stale = [tid for tid,t in self.tracks.items() if t.miss > self.max_age]
        for tid in stale:
            self.tracks.pop(tid, None)

        return assigned_tracks

# ===== PHÁT HIỆN TRẢ RA DANH SÁCH ==== #
def detect_objects(timeout_s=5.0, model_path="best.pt"):
    """
    Trả về list[dict]: mỗi dict có
      { 'tid', 'label', 'cls_id', 'conf', 'x_mm','y_mm','z_mm' }
    (label giữ nguyên đúng như YOLO)
    """
    cam = RealsenseCamera()
    intr = cam.get_intrinsics()
    model = YOLO(model_path)
    tracker = CentroidTracker(max_distance=60, max_age=15)

    meta = {}   # tid -> {'label','cls_id','conf'}
    coords = {} # tid -> {'x_mm','y_mm','z_mm'}

    start = time.time()
    cv2.namedWindow("Detect+Depth", cv2.WINDOW_NORMAL)

    while True:
        if (time.time() - start) >= timeout_s:
            break

        ret, bgr, depth_mm = cam.get_frame_stream()
        if not ret:
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        results = model(bgr)[0]
        detections, bboxes = [], []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = max(0, min((x1+x2)//2, bgr.shape[1]-1))
            cy = max(0, min((y1+y2)//2, bgr.shape[0]-1))
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)
            label = str(name)  # GIỮ NGUYÊN

            d_mm = median_depth_mm(depth_mm, cx, cy, k=5)
            detections.append({'cx':cx, 'cy':cy, 'd_mm':d_mm})
            bboxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                           'cls_id':cls_id,'label':label,'conf':conf,'cx':cx,'cy':cy})

        assigned = tracker.update(detections)

        for det, trk, bb in zip(detections, assigned, bboxes):
            if trk is None: continue
            tid = trk.tid
            meta[tid] = {'label': bb['label'], 'cls_id': bb['cls_id'], 'conf': bb['conf']}

            d_mm_avg = trk.avg_mm() if trk.avg_mm()>0 else det['d_mm']
            if d_mm_avg > 0:
                Z_m = d_mm_avg/1000.0
                X_m, Y_m, Z_m = rs.rs2_deproject_pixel_to_point(intr, [float(bb['cx']), float(bb['cy'])], Z_m)
                coords[tid] = {'x_mm': X_m*1000.0, 'y_mm': Y_m*1000.0, 'z_mm': Z_m*1000.0}

        # Hiển thị
        vis = bgr.copy()
        for bb, trk in zip(bboxes, assigned):
            cv2.rectangle(vis, (bb['x1'],bb['y1']), (bb['x2'],bb['y2']), (0,200,0), 2)
            if trk is not None:
                tid = trk.tid
                txt = f"{bb['label']}#{tid} {bb['conf']:.2f}"
                cv2.putText(vis, txt, (bb['x1'], max(20, bb['y1']-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                if tid in coords:
                    c = coords[tid]
                    cv2.putText(vis, f"X:{c['x_mm']:.0f} Y:{c['y_mm']:.0f} Z:{c['z_mm']:.0f}",
                                (bb['x1'], bb['y2']+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (50,200,255), 2, cv2.LINE_AA)

        cv2.imshow("Detect+Depth", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

    out = []
    for tid, m in meta.items():
        if tid in coords:
            out.append({
                'tid': tid,
                'label': m['label'],     # giữ nguyên
                'cls_id': m['cls_id'],   # giữ nguyên
                'conf': m['conf'],
                'x_mm': coords[tid]['x_mm'],
                'y_mm': coords[tid]['y_mm'],
                'z_mm': coords[tid]['z_mm'],
            })
    return out

# ==== Robot Move ==== #
def pick_objects(P_EE_cam, Rotation_xyz, speed=100, acc=20, tol=1):
    p = np.array([[P_EE_cam[0]], [P_EE_cam[1]], [P_EE_cam[2]], [1]])
    
    T = np.array([[1, 0, 0, 127],
                  [0,-1, 0, 101],
                  [0, 0,-1, 280],
                  [0, 0, 0,   1]])
    P_EE_1 = T @ p
    P_EE_2 = P_EE_1[:3,0]
    P_EE = np.hstack((P_EE_2, [math.radians(-180), math.radians(0), math.radians(-90)]))

    robot = jkrc.RC("192.168.31.15")
    ret = robot.login()
    if ret == 0:
        print("Login OK")

    robot.power_on()
    robot.enable_robot()

    home = [160, 25, 200, math.radians(-180), math.radians(0), math.radians(-90)]
    robot.linear_move(home, 0, False, speed)
    time.sleep(1.0)

    ret, = robot.linear_move_extend(P_EE, 0, False, speed, acc, tol)
    print("Go target ret =", ret)

    robot.linear_move_extend(home, 0, False, speed, acc, tol)
    robot.logout()
    return ret

# ==== Di chuyển robot tới điểm ==== #
def move_to_targets(detections, speed=100, acc=20, tol=1):
    """
    detections: list dict từ detect_objects()
    Chạy theo thứ tự ưu tiên: heart -> circle -> square.
    Với mỗi hình, đi từ gần (Z nhỏ) đến xa.
    """
    if not detections:
        print("Không có vật nào.")
        return

    total = 0
    for lab in PRIORITY:
        group = [d for d in detections if d['label'] == lab]
        group.sort(key=lambda x: x['z_mm'])  # gần trước
        for d in group:
            print(f"Đi tới {lab} (tid={d['tid']}): "
                  f"X={d['x_mm']:.0f} Y={d['y_mm']:.0f} Z={d['z_mm']:.0f} (conf={d['conf']:.2f})")
            ret = pick_objects(np.array([d['x_mm'], d['y_mm'], d['z_mm']]),
                               [math.radians(-180), math.radians(0), math.radians(-90)],
                               speed=speed, acc=acc, tol=tol)
            if ret == 0:
                print("→ DONE")
            else:
                print(f"→ LỖI (ret={ret})")
            total += 1

    if total == 0:
        print("Có phát hiện nhưng không khớp ưu tiên (heart/circle/square).")

# ==== MAIN ==== #
if __name__ == "__main__":
    objs = detect_objects(timeout_s=5.0, model_path="best.pt")
    print("Kết quả phát hiện:")
    for o in objs:
        print(f"  {o['label']}#{o['tid']}  conf={o['conf']:.2f}  "
              f"X={o['x_mm']:.0f}  Y={o['y_mm']:.0f}  Z={o['z_mm']:.0f}")

    move_to_targets(objs, speed=100, acc=20, tol=1)
