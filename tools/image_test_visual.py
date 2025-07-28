import os
import cv2
import json


def label_dictToMask(label_data, width=1280, height=720):
    # Start with a black image
    mask = np.zeros((height, width), dtype=np.uint8)

    lanes = label_data['lanes']
    h_samples = label_data['h_samples']

    # Go through each lane
    for lane in lanes:
        # Pair valid (x, y) points
        points = [(x, y) for x, y in zip(lane, h_samples) if x != -2 and 0 <= x < width and 0 <= y < height]
        

        # Draw lane line if there are enough points
        for i in range(1, len(points)):
            cv2.line(mask, points[i-1], points[i], 255, thickness=2)  # 255 = white
    return mask

with open("256x144/label_data_0313_256x144.json", 'r') as file:
    for _ in range(20):
        line = file.readline()
        if not line:
            break  # stop if file has less than 20 lines

        data = json.loads(line)

        # Get the mask
        mask = label_dictToMask(data, 256, 144)
        cv2.imshow("mask", mask)

        # Load and show the image
        img_path = os.path.join("256x144", data["raw_file"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        cv2.imshow("original_image", img)

        # Wait until key press before moving to next
        key = cv2.waitKey(0)
        if key == 27:  # Esc key to break early
            break

    cv2.destroyAllWindows()

