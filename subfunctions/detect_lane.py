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
    locate_important_edges(edges_of_interest)

    # Get coordinates of non-zero pixels
    y_coords, x_coords = np.nonzero(edges_of_interest)
    

    # Separate points to left and right of midpoint
    left_mask = x_coords < midpoint[0]
    right_mask = x_coords >= midpoint[0]
    
    left_x = x_coords[left_mask]
    left_y = y_coords[left_mask]
    right_x = x_coords[right_mask]
    right_y = y_coords[right_mask]
    
    # Polynomial regression for left and right sides using np.polyfit
    left_coeffs = np.polyfit(left_y, left_x, 2) if len(left_x) > 0 else None
    right_coeffs = np.polyfit(right_y, right_x, 2) if len(right_x) > 0 else None
    
    # Generate points for the regression lines - only for the bottom portion
    y_points = np.arange(int(height*0.62), height)
        
    # Calculate predicted x values using np.poly1d
    if left_coeffs is not None:
        left_predicted = np.poly1d(left_coeffs)(y_points)
    else:
        left_predicted = np.zeros_like(y_points)
        
    if right_coeffs is not None:
        right_predicted = np.poly1d(right_coeffs)(y_points)
    else:
        right_predicted = np.zeros_like(y_points)
    
    # Draw the regression lines in blue
    for i in range(len(y_points)-1):
        if 0 <= left_predicted[i] < width and 0 <= left_predicted[i+1] < width:
            cv2.line(result_image, 
                     (int(left_predicted[i]), y_points[i]), 
                     (int(left_predicted[i+1]), y_points[i+1]), 
                     (255, 0, 0), 2)  # Blue color
        
        if 0 <= right_predicted[i] < width and 0 <= right_predicted[i+1] < width:
            cv2.line(result_image, 
                     (int(right_predicted[i]), y_points[i]), 
                     (int(right_predicted[i+1]), y_points[i+1]), 
                     (255, 0, 0), 2)  # Blue color
    
    # Draw the path line in green
    for i in range(len(y_points)-1):
        x_avg_1 = int((left_predicted[i] + right_predicted[i]) / 2)
        x_avg_2 = int((left_predicted[i+1] + right_predicted[i+1]) / 2)
        
        if 0 <= x_avg_1 < width and 0 <= x_avg_2 < width:
            cv2.line(result_image, 
                     (x_avg_1, y_points[i]), 
                     (x_avg_2, y_points[i+1]), 
                     (0, 255, 0), 2)  # Green color
    
    # cv2.imshow("final_image", result_image) 
    cv2.waitKey(0)