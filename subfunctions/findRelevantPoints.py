import cv2
import numpy as np
def findRelativeWhitePoints(image, show_visualization=True):
    height, width = image.shape[:2]
    midpoint = np.array([int(width*0.48), int(height*0.6)])
    endpoint = np.array([width*0.24, height])
    
    # Create blank image for visualization
    visualization = np.zeros_like(image)
    
    edge_points = []
    for y in range(height):
        for x in range(width):
            if image[y, x] == 255:
                if point_to_line_distance(np.array([x, y]), midpoint, endpoint) <= 40:
                    edge_points.append((x, y))
                    # Add point to visualization
                    visualization[y, x] = 255
    
    print(f"Total edge points found: {len(edge_points)}")
    
    if show_visualization:
        # Draw the reference line
        cv2.line(visualization, 
                 (int(midpoint[0]), int(midpoint[1])), 
                 (int(endpoint[0]), int(endpoint[1])), 
                 255, 2)
        
        # Just show the visualization
        cv2.imshow('Filtered Edges', visualization)
  
    return edge_points, visualization

def point_to_line_distance(point, line_point1, line_point2):
    """
    Calculate the shortest distance from a point to a line defined by two points.
    """
    point = np.array(point)
    line_point1 = np.array(line_point1)
    line_point2 = np.array(line_point2)
    
    line_vec = line_point2 - line_point1
    point_vec = point - line_point1
    line_length = np.linalg.norm(line_vec)
    
    if line_length == 0:
        return np.linalg.norm(point_vec)
    
    line_unit_vec = line_vec / line_length
    projection_length = np.dot(point_vec, line_unit_vec)
    closest_point_vec = line_unit_vec * projection_length
    distance_vec = point_vec - closest_point_vec
    
    return np.linalg.norm(distance_vec)

# Example usage:
# Load and process your image
# image = cv2.imread('your_image.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edges = cv2.Canny(blurred, 20, 50)
# edge_points, visualization = findRelativeWhitePoints(edges)