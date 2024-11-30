import numpy as np
import cv2
import matplotlib.image as mpimg
from subfunctions.get_frame import get_frame_at_time
from subfunctions.detect_lane import detect_curved_lanes

# Example:
frame = get_frame_at_time('images/californiaLanes.mp4', 13)  # Gets frame at 5 seconds
detectedimage = detect_curved_lanes(frame)

# Clean up windows
cv2.destroyAllWindows()
