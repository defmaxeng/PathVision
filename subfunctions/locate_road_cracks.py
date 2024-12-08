import numpy as np
from typing import List, Tuple

def detect_dark_spots(image: np.ndarray, sensitivity: float = 5) -> List[Tuple[int, int]]:
    """
    Detect dark spots in a grayscale image by analyzing brightness gradients.
    
    Args:
        image: 2D numpy array representing a grayscale image (0-255)
        sensitivity: threshold for rate of change to detect darkness (default 0.1)
        
    Returns:
        List of (x,y) coordinates of pixels within dark spots
    """
    height, width = image.shape
    dark_pixels = []
    
    for y in range(height):
        in_dark_spot = False
        last_gradient = 0
        
        for x in range(1, width):
            # Calculate rate of change in brightness
            current_gradient = float(image[y,x].astype(int) - image[y,x-1].astype(int))            
            # Detecting start of dark spot (negative gradient exceeding sensitivity)
            if not in_dark_spot and current_gradient < -sensitivity:
                in_dark_spot = True
                last_gradient = current_gradient
            
            # Detecting end of dark spot (positive gradient similar to entry gradient)
            elif in_dark_spot and current_gradient > abs(last_gradient) - sensitivity:
                in_dark_spot = False
                
            # Save coordinates if we're in a dark spot
            if in_dark_spot:
                dark_pixels.append((x, y))
                
    return dark_pixels

def visualize_dark_spots(image: np.ndarray, dark_pixels: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create a visualization of detected dark spots.
    
    Args:
        image: Original grayscale image
        dark_pixels: List of coordinates in dark spots
        
    Returns:
        Copy of original image with dark spots highlighted
    """
    visualization = image.copy()
    
    # Mark dark pixels with white (255)
    for x, y in dark_pixels:
        visualization[y,x] = 255
        
    return visualization

# Example usage:
"""
import cv2

# Load grayscale image
image = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)

# Detect dark spots
dark_spots = detect_dark_spots(image, sensitivity=0.15)

# Visualize results
result = visualize_dark_spots(image, dark_spots)
cv2.imwrite('result.png', result)
"""