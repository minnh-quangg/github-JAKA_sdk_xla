# số ổn định hơn
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from realsense_camera import *

rs = RealsenseCamera()
model = YOLO("best.pt")
cv2.namedWindow("Measure Distance (mm)", cv2.WINDOW_NORMAL)

def median_depth_mm(depth_img, cx, cy, k=5):
    """Median kxk quanh (cx, cy); depth_img đơn vị mm."""
    h, w = depth_img.shape[:2]
    r = k // 2
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r + 1, w)
    y2 = min(cy + r + 1, h)
    roi = depth_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0
    return int(np.median(valid))

class Track:
    __slots__ = ("tid", "cx", "cy", "age", "miss", "depth_hist")
    def __init__(self, tid, cx, cy):
        self.tid = tid
        self.cx, self.cy = cx, cy
        self.age = 0       
        self.miss = 0      
        self.depth_hist = deque(maxlen=20)  # mm

    def update(self, cx, cy, d_mm):
        self.cx, self.cy = cx, cy
        self.age += 1
        self.miss = 0
        if d_mm > 0:
            self.depth_hist.append(d_mm)

    def predict_miss(self):
        self.miss += 1

    def avg_mm(self):
        return int(np.mean(self.depth_hist)) if len(self.depth_hist) > 0 else 0

class CentroidTracker:
    def __init__(self, max_distance=60, max_age=30):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}     
        self.next_id = 1

    @staticmethod
    def _dist(a, b):
        ax, ay = a; bx, by = b
        dx = ax - bx; dy = ay - by
        return (dx*dx + dy*dy) ** 0.5

    def update(self, detections):
        det_centers = [(d['cx'], d['cy']) for d in detections]
        tids = list(self.tracks.keys())
        tr_centers = [(self.tracks[tid].cx, self.tracks[tid].cy) for tid in tids]

        assigned_tracks = [None] * len(detections)
        if not tr_centers:
            for i, d in enumerate(detections):
                t = Track(self.next_id, d['cx'], d['cy'])
                t.update(d['cx'], d['cy'], d['d_mm'])
                self.tracks[self.next_id] = t
                assigned_tracks[i] = t
                self.next_id += 1
            return assigned_tracks

        D = np.zeros((len(tr_centers), len(det_centers)), dtype=np.float32)
        for i, tc in enumerate(tr_centers):
            for j, dc in enumerate(det_centers):
                D[i, j] = self._dist(tc, dc)

        used_tracks = set()
        used_dets = set()
        for _ in range(min(len(tr_centers), len(det_centers))):
            min_val = 1e9; min_i = -1; min_j = -1
            for i in range(len(tr_centers)):
                if i in used_tracks: continue
                for j in range(len(det_centers)):
                    if j in used_dets: continue
                    if D[i, j] < min_val:
                        min_val = D[i, j]; min_i = i; min_j = j
            if min_i == -1: break
            if min_val <= self.max_distance:
                tid = tids[min_i]
                t = self.tracks[tid]
                d = detections[min_j]
                t.update(d['cx'], d['cy'], d['d_mm'])
                assigned_tracks[min_j] = t
                used_tracks.add(min_i)
                used_dets.add(min_j)
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

        for i, tid in enumerate(tids):
            if i not in used_tracks:
                self.tracks[tid].predict_miss()

        stale = [tid for tid, t in self.tracks.items() if t.miss > self.max_age]
        for tid in stale:
            self.tracks.pop(tid, None)

        return assigned_tracks

tracker = CentroidTracker(max_distance=60, max_age=30)

while True:
    ret, bgr, depth_mm = rs.get_frame_stream()
    if not ret:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    frame = bgr.copy()
    results = model(frame)[0]

    detections = []
    bboxes = []  

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        d_mm_med = median_depth_mm(depth_mm, cx, cy, k=5) 

        detections.append({'cx': cx, 'cy': cy, 'd_mm': d_mm_med})
        bboxes.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cls': int(box.cls[0]),
            'conf': float(box.conf[0])
        })

    assigned_tracks = tracker.update(detections)

    # Draw
    for det, trk, bb in zip(detections, assigned_tracks, bboxes):
        x1, y1, x2, y2 = bb['x1'], bb['y1'], bb['x2'], bb['y2']
        cx, cy = det['cx'], det['cy']
        cls_id = bb['cls']
        conf = bb['conf']
        label = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)

        d_mm_med = det['d_mm']
        d_mm_avg = trk.avg_mm() if trk is not None else d_mm_med
        delta = 264 - d_mm_avg if d_mm_avg > 0 else None

        # bbox + center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # lines of text
        t1 = f"{label} {conf:.2f}"
        # t2 = f"d_med={d_mm_med}mm" if d_mm_med > 0 else "d_med=N/A"
        # t3 = f"d_avg20={d_mm_avg}mm" if d_mm_avg > 0 else "d_avg20=N/A"
        t4 = f"dis = {delta}mm" if delta is not None else "dis = N/A"
        if trk is not None:
            t1 = f"ID{trk.tid} | " + t1

        # pick text position outside bbox (above if possible)
        base_x = x1
        top_y = y1 - 10
        if top_y > 60:
            ys = [top_y, top_y-22, top_y-44, top_y-66]
        else:
            base = y2 + 20
            ys = [base, base+22, base+44, base+66]

        def put(txt, y, color_fg=(255,255,255), color_bg=(0,0,0)):
            cv2.putText(frame, txt, (base_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bg, 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (base_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_fg, 1, cv2.LINE_AA)

        put(t1, ys[0], (255,255,255))
        # put(t2, ys[1], (50,230,50))
        # put(t3, ys[2], (255,255,0))
        put(t4, ys[1], (50,200,255))

    cv2.imshow("Measure Distance (mm)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
