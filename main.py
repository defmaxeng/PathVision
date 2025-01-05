import numpy as np
import cv2
import matplotlib.image as mpimg
from subfunctions.get_frame import get_frame_at_time
from subfunctions.detect_lane import detect_curved_lanes
from process_video import process_the_video

input_video = "californiaLanes.mp4"
output_video = "californiaLanes_guided.mp4"
process_the_video(input_video, output_video)