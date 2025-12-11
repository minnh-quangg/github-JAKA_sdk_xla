from ultralytics import YOLO
import cv2, numpy as np
from collections import deque
from realsense_camera import *

# === Khởi tạo camera & model ===
rs = RealsenseCamera()
model = YOLO("best.pt")
cv2.namedWindow("Measure Distance (mm)", cv2.WINDOW_NORMAL)

# === Đo khoảng cách trung vị quanh tâm bbox ===
def median_depth_mm(depth, cx, cy, k=5):
    h, w = depth.shape[:2]; r = k // 2
    x1, y1, x2, y2 = max(cx-r,0), max(cy-r,0), min(cx+r+1,w), min(cy+r+1,h)
    roi = depth[y1:y2, x1:x2]
    valid = roi[roi > 0]
    return int(np.median(valid)) if valid.size else 0

# === Tracking thủ công ===
tracks, next_id = {}, 1
MAX_DISTANCE, MAX_AGE = 60, 30

def create_track(cx, cy, d):
    global next_id
    t = {"tid": next_id, "cx": cx, "cy": cy, "age": 0, "miss": 0,
         "depth_hist": deque([d] if d > 0 else [], maxlen=20)}
    next_id += 1
    return t

def update_track(t, cx, cy, d):
    t["cx"], t["cy"], t["age"], t["miss"] = cx, cy, t["age"]+1, 0
    if d > 0: t["depth_hist"].append(d)

def avg_mm(t): return int(np.mean(t["depth_hist"])) if t["depth_hist"] else 0
def distance(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])

def update_tracks(detections):
    global tracks
    tids = list(tracks.keys())
    tr_centers = [(tracks[tid]["cx"], tracks[tid]["cy"]) for tid in tids]
    assigned = [None]*len(detections)
    if not tr_centers:
        for i,d in enumerate(detections):
            t = create_track(d["cx"], d["cy"], d["d_mm"])
            tracks[t["tid"]] = assigned[i] = t
        return assigned

    # Ma trận khoảng cách
    D = np.array([[distance(tc,(d["cx"],d["cy"])) for d in detections] for tc in tr_centers])
    used_t, used_d = set(), set()
    for _ in range(min(len(tr_centers), len(detections))):
        i,j = divmod(np.argmin(D), D.shape[1])
        if D[i,j] > MAX_DISTANCE: break
        tid, t, d = tids[i], tracks[tids[i]], detections[j]
        update_track(t, d["cx"], d["cy"], d["d_mm"])
        assigned[j], D[i,:], D[:,j] = t, np.inf, np.inf
        used_t.add(i); used_d.add(j)

    # Tạo track mới & cập nhật miss
    for j,d in enumerate(detections):
        if j not in used_d:
            t = create_track(d["cx"], d["cy"], d["d_mm"])
            tracks[t["tid"]] = assigned[j] = t
    for i,tid in enumerate(tids):
        if i not in used_t: tracks[tid]["miss"] += 1
    for tid in [tid for tid,t in tracks.items() if t["miss"]>MAX_AGE]:
        tracks.pop(tid)
    return assigned

def put(frame, txt, pos, fg=(255,255,255), bg=(0,0,0)):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)

# === Vòng lặp chính ===
while True:
    ret, bgr, depth = rs.get_frame_stream()
    if not ret or cv2.waitKey(1)&0xFF==27: break

    frame = bgr.copy()
    results = model(frame)[0]
    detections, bboxes = [], []

    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cx,cy = (x1+x2)//2, (y1+y2)//2
        d = median_depth_mm(depth,cx,cy)
        detections.append({"cx":cx,"cy":cy,"d_mm":d})
        bboxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                       "cls":int(box.cls[0]),"conf":float(box.conf[0])})

    tracks_assigned = update_tracks(detections)

    for det,trk,bb in zip(detections, tracks_assigned, bboxes):
        x1,y1,x2,y2 = bb["x1"],bb["y1"],bb["x2"],bb["y2"]
        cx,cy = det["cx"],det["cy"]
        label = model.names.get(bb["cls"], str(bb["cls"])) if hasattr(model,"names") else str(bb["cls"])
        d_avg = avg_mm(trk) if trk else det["d_mm"]
        delta = f"{277-d_avg}mm" if d_avg>0 else "N/A"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,210,0),2)
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        put(frame,f"{label} {bb['conf']:.2f}",(x1,max(20,y1-10)))
        put(frame,f"dis = {delta}",(x1,max(40,y1+20)),(50,200,255))

    cv2.imshow("Measure Distance (mm)",frame)

cv2.destroyAllWindows()
