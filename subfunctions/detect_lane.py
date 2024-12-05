import cv2
import numpy as np
from subfunctions.findRelevantPoints import findRelativeWhitePoints


def detect_curved_lanes(image):
    """
    Detect lanes using linear regression for bottom 50% and polynomial for upper part
    """
    height, width = image.shape[:2]
    midpoint = (int(width*0.48), int(height*0.6))
    # Preprocessing steps
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 50)
    findRelativeWhitePoints(edges)
    cv2.line(edges, (int(width/2.1), height), (int(width/2.1), 0), (255, 0, 0), 1)
    
    cv2.imshow('Canny Edge Detection', edges)

    # Create ROI mask
    roi_vertices = np.array([
        [(width/33, height),
         (midpoint[0] - 70, height//1.8 + 10),
         (midpoint[0] + 70, height//1.8 + 10),
         (width*3/3, height)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    cv2.imshow('brfuh', masked_edges)
    roi_display = image.copy()
    # Draw filled polygon
    overlay = roi_display.copy()
    cv2.fillPoly(overlay, roi_vertices, (0, 255, 0))  # Green color
    # Blend with original
    alpha = 0.3
    roi_display = cv2.addWeighted(overlay, alpha, roi_display, 1 - alpha, 0)
    # Draw ROI boundaries
    cv2.polylines(roi_display, roi_vertices, True, (255, 0, 0), 2)  # Red boundary
    
    # Display ROI
    cv2.imshow('Region of Interest', roi_display)
    # Find all non-zero points in the masked edge image
    nonzero = masked_edges.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Separate points into left and right lane based on position
    midpoint = width // 2
    left_lane_points = (nonzerox < midpoint)
    right_lane_points = (nonzerox >= midpoint)
    
    leftx = nonzerox[left_lane_points]
    lefty = nonzeroy[left_lane_points]
    rightx = nonzerox[right_lane_points]
    righty = nonzeroy[right_lane_points]
    
    # Split points into bottom 50% and upper part
    bottom_cutoff = int(height * 0.65)  # Changed from 0.75 to 0.5
    
    # Left lane splits
    left_bottom_mask = lefty >= bottom_cutoff
    left_top_mask = lefty < bottom_cutoff
    
    left_bottom_x = leftx[left_bottom_mask]
    left_bottom_y = lefty[left_bottom_mask]
    left_top_x = leftx[left_top_mask]
    left_top_y = lefty[left_top_mask]
    
    # Right lane splits
    right_bottom_mask = righty >= bottom_cutoff
    right_top_mask = righty < bottom_cutoff
    
    right_bottom_x = rightx[right_bottom_mask]
    right_bottom_y = righty[right_bottom_mask]
    right_top_x = rightx[right_top_mask]
    right_top_y = righty[right_top_mask]
    
    # Generate points for drawing
    result = image.copy()
    
    # Fit lines if we have points
    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
        # Linear fit for bottom 50%
        if len(left_bottom_y) > 0:
            left_bottom_fit = np.polyfit(left_bottom_y, left_bottom_x, 1)
        if len(right_bottom_y) > 0:
            right_bottom_fit = np.polyfit(right_bottom_y, right_bottom_x, 1)
            
        # Polynomial fit for upper part
        if len(left_top_y) > 0:
            left_top_fit = np.polyfit(left_top_y, left_top_x, 2)
        if len(right_top_y) > 0:
            right_top_fit = np.polyfit(right_top_y, right_top_x, 2)
        
        # Generate points for bottom part (linear)
        plot_y_bottom = np.linspace(bottom_cutoff, height-1, 20)
        left_bottom_fitx = left_bottom_fit[0]*plot_y_bottom + left_bottom_fit[1]
        right_bottom_fitx = right_bottom_fit[0]*plot_y_bottom + right_bottom_fit[1]
        
        # Generate points for top part (polynomial)
        plot_y_top = np.linspace(0, bottom_cutoff-1, 40)
        left_top_fitx = left_top_fit[0]*plot_y_top**2 + left_top_fit[1]*plot_y_top + left_top_fit[2]
        right_top_fitx = right_top_fit[0]*plot_y_top**2 + right_top_fit[1]*plot_y_top + right_top_fit[2]
        
        # Combine points
        left_fitx = np.concatenate([left_top_fitx, left_bottom_fitx])
        right_fitx = np.concatenate([right_top_fitx, right_bottom_fitx])
        plot_y = np.concatenate([plot_y_top, plot_y_bottom])
        
        # Convert to points for drawing
        left_points = np.array([np.transpose(np.vstack([left_fitx, plot_y]))], dtype=np.int32)
        right_points = np.array([np.transpose(np.vstack([right_fitx, plot_y]))], dtype=np.int32)
        
        # Draw the lanes
        cv2.polylines(result, [left_points], False, (0, 255, 0), 2)
        cv2.polylines(result, [right_points], False, (0, 255, 0), 2)
        
        # Create an overlay for the filled area
        overlay = result.copy()
        
        # Create points for the polygon
        pts = np.vstack((
            left_points[0],
            np.flip(right_points[0], axis=0)
        ))
        pts = pts.reshape((-1, 1, 2))
        
        # Fill the polygon
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        
        # Blend the overlay with the original image
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # Draw a line showing the split between linear and polynomial regions
        cv2.line(result, (0, bottom_cutoff), (width, bottom_cutoff), (255, 0, 0), 1)
        cv2.line(result, (int(width*0.48), height), (int(width*0.48), 0), (255, 0, 0), 1)
        cv2.line(result, (0, int(height*0.6)), (width, int(height*0.6)), (255, 0, 0), 1)

    # Display the result
    cv2.imshow('Lane Detection', result)
    cv2.waitKey(0)
    
    return result
