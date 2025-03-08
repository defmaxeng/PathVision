import numpy as np
import cv2

def locate_important_edges(canny_image):
    """
    Takes in: Cannyed Image in numpy array with all edges post mask application
    Returns: Lists of regression parameters (slope, intercept) for left and right sides
    
    Implementation of sliding window linear regression using np.polyfit:
    1. Separates points into left and right sides based on midline
    2. Performs linear regression on vertical windows
    3. Stores regression parameters for each window
    """
    global regyes
    regyes = 0
    global regno
    regno = 0
    # Variable declarations
    height, width = canny_image.shape[:2]
    highest_window_startpos = int(height * 0.5)
    windowsize = 15
    approx_midline = int(width * 0.46)
    
    # Arrays to store regression parameters (slope, intercept) for each side
    left_regressions = []
    right_regressions = []
    left_y, left_x, right_y, right_x, extreme_image = middle_xpoints(canny_image)

    def window_regression(x_points, y_points, window_start, window_end):
        global regno
        global regyes
        # Find points within the window
        window_indices = (y_points >= window_start) & (y_points < window_end)
        window_x = x_points[window_indices]
        window_y = y_points[window_indices]
        
        # If we have enough points, perform regression
        if len(window_x) >= 2:  # Need at least 2 points for regression
            
            # polyfit returns coefficients [slope, intercept]
            coeffs = np.polyfit(window_x, window_y, 1)
            regyes += 1

            return (coeffs[0], coeffs[1])  # slope, intercept
        
        else:
            regno += 1
        return None
    
    # Sliding window regression for left side
    for window_start in range(height - windowsize, int(highest_window_startpos), -windowsize):
        window_end = window_start + windowsize
        regression_params = window_regression(left_x, left_y, window_start, window_end)
        if regression_params is not None:
            left_regressions.append(regression_params)
    
    # Sliding window regression for right side
    for window_start in range(height - windowsize, int(highest_window_startpos), -windowsize):
        window_end = window_start + windowsize
        regression_params = window_regression(right_x, right_y, window_start, window_end)
        if regression_params is not None:
            right_regressions.append(regression_params)
    
    visualize_slopes_and_lines(canny_image, left_regressions, right_regressions, 10)

    regression_image = cv2.cvtColor(extreme_image, cv2.COLOR_GRAY2BGR)

    # cv2.imshow('regression', regression_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    

    # get_midline(find_most_common_slope(left_regressions), find_most_common_slope(right_regressions), height, width, extreme_image)

    return get_midline(find_most_common_line(left_regressions), find_most_common_line(right_regressions), height, extreme_image)

def get_midline(left_line, right_line, height, extreme_image):
    """
    Calculates and displays the midline based on the average x-coordinates
    of the left and right lanes at each row. Only returns points with y > 300.
    
    Parameters:
    left_line: (slope, intercept) tuple for the left line
    right_line: (slope, intercept) tuple for the right line
    height: Height of the image
    extreme_image: Grayscale image for visualization
    """
    # Unpack slope and intercept for both lines
    left_slope, left_intercept = left_line
    right_slope, right_intercept = right_line

    # Create midline image
    midline_image = cv2.cvtColor(extreme_image, cv2.COLOR_GRAY2BGR)

    # Generate midline points
    midline_points = []
    for y in range(height):
        # Only process points with y > 300
        if y > 500:
            # Calculate x-coordinates for left and right lines at the current row (y)
            left_x = (y - left_intercept) / left_slope
            right_x = (y - right_intercept) / right_slope

            # Average the x-coordinates to find the midline point
            mid_x = int((left_x + right_x) / 2)

            # Add point to list
            midline_points.append((mid_x, y))

    # Draw the midline on the image (only if there are at least 2 points)
    if len(midline_points) > 1:
        for i in range(len(midline_points) - 1):
            cv2.line(midline_image, midline_points[i], midline_points[i + 1], (0, 255, 0), 2)

    return midline_points

def find_most_common_line(lines, threshold=0.3):
    """
    Finds the most common slope from a list of slopes
    
    Parameters:
    slopes: List of slope values
    threshold: Minimum difference between slopes to be considered different
    
    Returns:
    Most common slope
    """
    slopes = [slope[0] for slope in lines]

    # Initialize dictionary to store counts
    slope_counts = {}
    
    # Count occurrences of each slope
    for slope in slopes:
        found = False
        for key in slope_counts:
            if abs(slope - key) < threshold:
                slope_counts[key] += 1
                found = True
                break
        if not found:
            slope_counts[slope] = 1
    
    # Find the most common slope
    most_common_slope = max(slope_counts, key=slope_counts.get)

    intercepts = [intercept[1] for intercept in lines]

    # Initialize dictionary to store counts
    intercept_counts = {}
    
    # Count occurrences of each slope
    for intercept in intercepts:
        found = False
        for key in intercept_counts:
            if abs(intercept - key) < threshold:
                intercept_counts[key] += 1
                found = True
                break
        if not found:
            intercept_counts[intercept] = 1
    
    # Find the most common slope
    most_common_intercept = max(intercept_counts, key=intercept_counts.get)
    # print(most_common_slope, most_common_intercept)
    return most_common_slope, most_common_intercept

import numpy as np

def middle_xpoints(canny_image):
    """
    Takes in a Canny image and returns:
    1. Arrays of x and y coordinates of first non-zero point in each row
    2. A binary image showing only these points
    """
    height, width = canny_image.shape[:2]
    first_x_points = []
    first_y_points = []

    last_x_points = []
    last_y_points = []
    approx_midline = int(width * 0.46)

    
    
    # Create empty black image
    point_image = np.zeros_like(canny_image)

    # Start from 65% of height to bottom
    for row in range(int(height * 0.65), height):
        for column in range(approx_midline, 0, -1):
            if canny_image[row, column] != 0:
                first_x_points.append(row)
                first_y_points.append(column)
                # Set the point in our new image to white (255)
                point_image[row, column] = 255
                break

    for row in range(int(height * 0.65), height):
        for column in range(approx_midline, width):
            if canny_image[row, column] != 0:
                last_x_points.append(row)
                last_y_points.append(column)
                # Set the point in our new image to white (255)
                point_image[row, column] = 255
                break


    
    return (np.array(first_x_points), 
        np.array(first_y_points), 
        np.array(last_x_points), 
        np.array(last_y_points),
        point_image
    )

import matplotlib.pyplot as plt

def visualize_slopes_and_lines(canny_image, left_regressions, right_regressions, bins=10):
    """
    Creates histograms of slope distributions and visualizes detected lines
    
    Parameters:
    canny_image: Original canny-transformed image
    left_regressions: List of (slope, intercept) tuples for left lane
    right_regressions: List of (slope, intercept) tuples for right lane
    bins: Number of bins for histogram
    """
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # Original image with lines subplot
    ax_img = fig.add_subplot(gs[:, 0])  # Left half of figure
    ax1 = fig.add_subplot(gs[0, 1])     # Top right
    ax2 = fig.add_subplot(gs[1, 1])     # Bottom right
    
    # Extract slopes
    left_slopes = [slope for slope, _ in left_regressions]
    right_slopes = [slope for slope, _ in right_regressions]
    
    # Plot original image with lines
    height, width = canny_image.shape[:2]
    ax_img.imshow(canny_image, cmap='gray')
    
    # Plot lines for each window
    for i, (slope, intercept) in enumerate(left_regressions):
        # Calculate window boundaries
        window_start = height - 15 * (i + 1)  # Using windowsize=15 from original code
        window_end = window_start + 15
        
        # Calculate x coordinates for this y range
        y = np.array([window_start, window_end])
        x = (y - intercept) / slope  # y = mx + b -> x = (y-b)/m
        
        # Plot only if within image bounds
        valid_points = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(valid_points):
            ax_img.plot(x[valid_points], y[valid_points], 'b-', linewidth=2)
    
    for i, (slope, intercept) in enumerate(right_regressions):
        window_start = height - 15 * (i + 1)
        window_end = window_start + 15
        
        y = np.array([window_start, window_end])
        x = (y - intercept) / slope
        
        valid_points = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(valid_points):
            ax_img.plot(x[valid_points], y[valid_points], 'g-', linewidth=2)
    
    ax_img.set_title('Detected Lane Lines')
    
    # Plot left slopes histogram
    ax1.hist(left_slopes, bins=bins, color='blue', alpha=0.7)
    ax1.set_title('Left Lane Slope Distribution')
    ax1.set_xlabel('Slope')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(left_slopes), color='r', linestyle='dashed', linewidth=1)
    ax1.text(0.02, 0.95, f'Mean: {np.mean(left_slopes):.3f}\nStd: {np.std(left_slopes):.3f}', 
             transform=ax1.transAxes, verticalalignment='top')
    
    # Plot right slopes histogram
    ax2.hist(right_slopes, bins=bins, color='green', alpha=0.7)
    ax2.set_title('Right Lane Slope Distribution')
    ax2.set_xlabel('Slope')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(right_slopes), color='r', linestyle='dashed', linewidth=1)
    ax2.text(0.02, 0.95, f'Mean: {np.mean(right_slopes):.3f}\nStd: {np.std(right_slopes):.3f}', 
             transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("Left Lane Slopes:")
    print(f"  Mean: {np.mean(left_slopes):.3f}")
    print(f"  Std:  {np.std(left_slopes):.3f}")
    print(f"  Min:  {np.min(left_slopes):.3f}")
    print(f"  Max:  {np.max(left_slopes):.3f}")
    
    print("\nRight Lane Slopes:")
    print(f"  Mean: {np.mean(right_slopes):.3f}")
    print(f"  Std:  {np.std(right_slopes):.3f}")
    print(f"  Min:  {np.min(right_slopes):.3f}")
    print(f"  Max:  {np.max(right_slopes):.3f}")