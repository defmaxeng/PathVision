import cv2
import numpy as np
from subfunctions.region_of_interest import apply_region_of_interest
from subfunctions.find_important_edges import locate_important_edges

def detect_curved_lanes(image):
    """
    Detect lanes using linear regression for bottom 50% and polynomial for upper part
    """
    height, width = image.shape[:2]
    midpoint = (int(width*0.46), int(height*0.6))
    
    # Create a copy of the original image for drawing
    result_image = image.copy()
    # Preprocessing steps
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 130)
    edges_of_interest = apply_region_of_interest(edges)
    pathline = locate_important_edges(edges_of_interest)
    
    # Draw the pathline points in green
    for point in pathline:
        # Convert tuple to integer coordinates
        x, y = int(point[0]), int(point[1])
        # Draw a small circle at each point
        cv2.circle(result_image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    
    # Optionally, connect the points with lines
    if len(pathline) > 1:
        # Convert pathline to numpy array of integers
        points = np.array(pathline, dtype=np.int32)
        # Draw lines connecting the points
        cv2.polylines(result_image, [points], isClosed=False, color=(0, 255, 0), thickness=2)
    
    return result_image
    

    