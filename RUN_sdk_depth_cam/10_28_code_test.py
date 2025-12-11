from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from realsense_camera import *
import pyrealsense2 as rs  # cần cho rs.rs2_deproject_pixel_to_point

# === Init ===
cam = RealsenseCamera()
intr = cam.get_intrinsics()           # intrinsics COLOR (align depth->color)
model = YOLO("best.pt")
cv2.namedWindow("Measure Distance (mm)", cv2.WINDOW_NORMAL)

# Median depth kxk quanh điểm
def median_depth_mm(depth_mm, cx, cy, k=5):
    h, w = depth_mm.shape[:2]
    r = k // 2
    x1 = max(cx - r, 0); y1 = max(cy - r, 0)
    x2 = min(cx + r + 1, w); y2 = min(cy + r + 1, h)
    roi = depth_mm[y1:y2, x1:x2]
    if roi.size == 0: return 0
    valid = roi[roi > 0]
    if valid.size == 0: return 0
    return float(np.median(valid))     # mm (float)

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
        return float(np.mean(self.depth_hist)) if len(self.depth_hist) > 0 else 0.0

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

tracker = CentroidTracker(max_distance=60, max_age=30)

def put(frame, txt, pos, fg=(255,255,255), bg=(0,0,0)):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)

while True:
    ret, bgr, depth_mm = cam.get_frame_stream()
    if not ret:
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    frame = bgr.copy()
    results = model(frame)[0]

    detections = []
    bboxes = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = max(0, min((x1 + x2)//2, frame.shape[1]-1))
        cy = max(0, min((y1 + y2)//2, frame.shape[0]-1))

        d_mm_med = median_depth_mm(depth_mm, cx, cy, k=5)  # mm
        detections.append({'cx': cx, 'cy': cy, 'd_mm': d_mm_med})
        bboxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                       'cls':int(box.cls[0]), 'conf':float(box.conf[0])})

    assigned = tracker.update(detections)

    for det, trk, bb in zip(detections, assigned, bboxes):
        x1,y1,x2,y2 = bb['x1'],bb['y1'],bb['x2'],bb['y2']
        cx, cy = det['cx'], det['cy']
        cls_id = bb['cls']; conf = bb['conf']
        label = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)

        # --- Depth ổn định (mm) ---
        d_mm_avg = trk.avg_mm() if trk is not None else det['d_mm']

        # --- Deproject bằng RealSense: Z theo mét ---
        if d_mm_avg > 0:
            Z_m = d_mm_avg / 1000.0
            pixel = [float(cx), float(cy)]
            X_m, Y_m, Z_m = rs.rs2_deproject_pixel_to_point(intr, pixel, Z_m)  # (m)
            X_mm, Y_mm, Z_mm = X_m*1000.0, Y_m*1000.0, Z_m*1000.0
            coord_line = f"{label}: X={X_mm:.0f}mm, Y={Y_mm:.0f}mm, Z={Z_mm:.0f}mm"
            t2 = f"dis = {271 - d_mm_avg:.0f}mm"
        else:
            coord_line = f"{label}: X=N/A, Y=N/A, Z=N/A"
            t2 = "dis = N/A"

        # --- Vẽ ---
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,210,0), 2)
        cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)

        base_x = x1
        top_y = y1 - 10
        if top_y > 60:
            ys = [top_y, top_y-22, top_y-44]
        else:
            base = y2 + 20
            ys = [base, base+22, base+44]

        put(frame, f"{label} {conf:.2f}", (base_x, ys[0]))
        put(frame, t2, (base_x, ys[1]), (50,200,255))
        put(frame, coord_line, (base_x, ys[2]), (0,255,255))

    cv2.imshow("Measure Distance (mm)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cam.release()
