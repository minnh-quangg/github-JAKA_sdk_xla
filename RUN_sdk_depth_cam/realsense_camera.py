import pyrealsense2 as rs
import numpy as np

class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # CHANGED: dùng COLOR intrinsics thay vì DEPTH intrinsics
        self.color_intr = self.profile.get_stream(rs.stream.color) \
            .as_video_stream_profile().get_intrinsics()   # ADDED

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_frame_stream(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        depth_raw = np.asanyarray(depth_frame.get_data())         # uint16
        # ADDED: đổi sang mm để code phía ngoài dùng thống nhất mm
        depth_mm = (depth_raw.astype(np.float32) * self.depth_scale * 1000.0).astype(np.float32)

        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image, depth_mm                          # CHANGED: trả depth_mm

    def get_intrinsics(self):
        return self.color_intr                                      # CHANGED: trả COLOR intrinsics

    def release(self):
        self.pipeline.stop()
