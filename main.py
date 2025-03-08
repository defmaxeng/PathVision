import numpy as np
import cv2
import matplotlib.image as mpimg
from subfunctions.get_frame import get_frame_at_time
from subfunctions.detect_lane import detect_curved_lanes
from process_video import process_this_video

input_video = "images/lane_video.mp4"
output_video = "californiaLanes_guided.mp4"

detect_curved_lanes(get_frame_at_time(input_video, 10))