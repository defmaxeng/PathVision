import json
import cv2
import os
from pathlib import Path

def process_images_with_canny(json_file, output_base_dir="clips_canny"):
    """
    Process images from JSON file and apply Canny edge detection.
    
    Args:
        json_file (str): Path to the JSON file containing image paths
        output_base_dir (str): Base directory for output (default: "clips_canny")
    """
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    print(f"Reading JSON file: {json_file}")
    
    # Read JSON file (assuming JSONL format - one JSON object per line)
    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Get the raw file path
                if 'raw_file' not in data:
                    print(f"Line {line_num}: No 'raw_file' field found")
                    continue
                    
                raw_file = data['raw_file']
                print(f"Processing: {raw_file}")
                raw_file = os.path.join("images/256x144", raw_file)
                # Check if source image exists
                if not os.path.exists(raw_file):
                    print(f"Warning: Source image not found: {raw_file}")
                    error_count += 1
                    continue
                
                # Create output path by replacing "clips" with "clips_canny"
                # e.g., "clips/0313-1/44580/20.jpg" -> "clips_canny/0313-1/44580/20.jpg"
                output_path = raw_file.replace("images/256x144_canny/clips", output_base_dir, 1)
                
                # Create output directory structure
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Read image
                image = cv2.imread(raw_file)
                if image is None:
                    print(f"Error: Could not read image: {raw_file}")
                    error_count += 1
                    continue
                
                # Convert to grayscale for Canny
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply Canny edge detection
                # You can adjust these thresholds (100, 200) as needed
                canny = cv2.Canny(gray, 100, 200)
                
                # Save the Canny edge image
                success = cv2.imwrite(output_path, canny)
                
                if success:
                    processed_count += 1
                    if processed_count % 100 == 0:  # Progress update every 100 images
                        print(f"Processed {processed_count} images...")
                else:
                    print(f"Error: Failed to save image: {output_path}")
                    error_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                error_count += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}")
                error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count}")
    print(f"Output directory: {output_base_dir}")

def process_single_json_object(json_file, output_base_dir="clips_canny"):
    """
    Alternative function if your JSON file contains a single JSON object 
    instead of JSONL format.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # If it's a single object with a raw_file
    if 'raw_file' in data:
        raw_file = data['raw_file']
        # Process single image (same logic as above)
        # ... implement if needed
    
    # If it's an array of objects
    elif isinstance(data, list):
        for item in data:
            if 'raw_file' in item:
                # Process each item
                pass

if __name__ == "__main__":
    # Configuration
    json_file_path = "images/256x144/label_data_0313_256x144.json"  # Replace with your JSON file path
    output_directory = "images/256x144_canny"        # Output directory name
    
    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        print("Please update the 'json_file_path' variable with the correct path.")
    else:
        # Process images
        process_images_with_canny(json_file_path, output_directory)
        
    # Optional: Adjust Canny parameters
    # You can modify the Canny thresholds in the cv2.Canny(gray, 100, 200) line
    # Lower threshold = more edges detected
    # Higher threshold = fewer edges detected