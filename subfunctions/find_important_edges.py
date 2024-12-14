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
    highest_window_startpos = int(height * 0.7)
    windowsize = 55
    approx_midline = int(width * 0.46)
    
    # Arrays to store regression parameters (slope, intercept) for each side
    left_regressions = []
    right_regressions = []
    left_x, left_y, right_x, right_y, extreme_image = extreme_xpoints(canny_image)

    testimage = np.zeros_like()
    for x, y in zip(left_x, left_y):
        testimage[y, x] = 255  # Note: OpenCV uses [y, x] ordering
    cv2.imshow("testimage", testimage)
    # Find all non-zero points (edge points)
    y_points, x_points = np.nonzero(canny_image)
    
    # Separate points into left and right sides
    left_indices = x_points < approx_midline
    right_indices = x_points >= approx_midline
    
    # left_x = x_points[left_indices]
    # left_y = y_points[left_indices]
    # right_x = x_points[right_indices]
    # right_y = y_points[right_indices]
    
    # Function to perform regression on a window of points using np.polyfit
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
            print("window: ", str(window_start)," to ", str(window_end)," Value: ", window_y)

            return (coeffs[0], coeffs[1])  # slope, intercept
        
        else:
            print("window: ", str(window_start)," to ", str(window_end), " nothing")
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
    
    visualize_slopes_and_lines(extreme_image, left_regressions, right_regressions, 10)
    print ("successful regressions: " + str(regyes) + "less successful: " + str(regno))
    return left_regressions, right_regressions

import numpy as np

def extreme_xpoints(canny_image):
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
        for column in range(0, approx_midline):
            if canny_image[row, column] != 0:
                first_x_points.append(row)
                first_y_points.append(column)
                # Set the point in our new image to white (255)
                point_image[row, column] = 255
                break

    for row in range(int(height * 0.65), height):
        for column in range(width-1, approx_midline, -1):
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