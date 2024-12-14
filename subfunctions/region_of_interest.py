import cv2
import numpy as np

def apply_region_of_interest(image):
    # establish variables
    height, width = image.shape[:2]
    midpoint = (int(width*0.46), int(height*0.64))
    margin_of_error = 120
    lower_left_vertex = (width*0.3, height)
    lower_right_vertex = (width-width*0.28, height)

    #Make shape
    roi_vertices = np.array([
        [(lower_left_vertex[0]-margin_of_error, lower_left_vertex[1]),
         (midpoint[0] - 0.5*margin_of_error, midpoint[1]),
         (midpoint[0] + 0.5*margin_of_error, midpoint[1]),
         (lower_right_vertex[0] + margin_of_error, lower_right_vertex[1]),
         (lower_right_vertex[0] - margin_of_error, lower_right_vertex[1]),
         (midpoint[0], midpoint[1]+margin_of_error),
         (lower_left_vertex[0]+margin_of_error, lower_left_vertex[1])
         ]
    ], dtype=np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(image, mask)
    
    # Create named window first
    # cv2.imshow('Region of Interest', masked_edges)
    roi_display = image.copy()
    overlay = roi_display.copy()
    cv2.fillPoly(overlay, roi_vertices, (0, 255, 0))
    alpha = 0.3
    roi_display = cv2.addWeighted(overlay, alpha, roi_display, 1 - alpha, 0)
    cv2.polylines(roi_display, roi_vertices, True, (125, 0, 0), 2)
    
    # cv2.imshow('maskshown', roi_display)

    return masked_edges

def find_Relative_White_Points(image, show_visualization=True):
    height, width = image.shape[:2]
    midpoint = np.array([int(width*0.48), int(height*0.6)])
    endpoint = np.array([width*0.24, height])
    
    visualization = np.zeros_like(image)
    edge_points = []
    
    for y in range(height):
        for x in range(width):
            if image[y, x] == 255:
                if point_to_line_distance(np.array([x, y]), midpoint, endpoint) <= 40:
                    edge_points.append((x, y))
                    visualization[y, x] = 255
    
    print(f"Total edge points found: {len(edge_points)}")
    
    if show_visualization:
        cv2.line(visualization, 
                 (int(midpoint[0]), int(midpoint[1])), 
                 (int(endpoint[0]), int(endpoint[1])), 
                 255, 2)
        
        # Create named window and set size for visualization
        cv2.namedWindow('Filtered Edges', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Filtered Edges', 800, 600)  # Adjust these values as needed
        # cv2.imshow('Filtered Edges', visualization)
  
    return edge_points, visualization

# Rest of the code remains the same...