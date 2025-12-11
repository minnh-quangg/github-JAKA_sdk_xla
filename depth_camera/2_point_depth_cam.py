import cv2
from realsense_camera import*

rs = RealsenseCamera()

points = [] #(x,y)

def click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) > 2:
            points.pop(0)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()

cv2.namedWindow("Bgr frame")
cv2.setMouseCallback("Bgr frame", click)

while True:
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    for i, (x, y) in enumerate (points):
        distance_mm = depth_frame[y, x]
        if i==0: 
            color = (0, 0, 255) # Đỏ
        else: 
            color = (0, 255, 0) # Xanh lá
        cv2.circle(bgr_frame, (x, y), 8, color, -1)
        cv2.putText(bgr_frame, "{}mm". format(distance_mm), (x, y-10), 0, 0.75, color, 2)
    
    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
