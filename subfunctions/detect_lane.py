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
    
    return locate_important_edges(edges_of_interest)
    