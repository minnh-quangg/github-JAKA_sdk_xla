import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

# Chuyển sang numpy
depth_image = np.asanyarray(depth_frame.get_data())

# Lưu ra file depth.png
cv2.imwrite("depth.png", depth_image)