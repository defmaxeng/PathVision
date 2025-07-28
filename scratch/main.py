from network_layers import Convolution, ReLU, Flatten, Dense, Reshape, Softmax, Model, MaxPool2D
from preprocessing import label_dictToMask
import cv2
import json
import numpy as np
import os


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



# print(f"archive/TUSimple/train_set/{image_path}")
# # img = cv2.imread(image_path)       # Loads image as NumPy array (H, W, C)

# depth, height, width = 3, 720, 1280  # Example input dimensions
# firstConvNetwerk = Model()
# # Convolution: Requires input_shape, kernel_size, kernel_count.
# firstConvNetwerk.add(Convolution(input_shape=(height, width, depth), kernel_size=3, kernel_count=3))

# # ReLU: No parameters required.
# firstConvNetwerk.add(ReLU())

# # MaxPool2D: Requires size and stride.
# firstConvNetwerk.add(MaxPool2D(size=2, stride=2))

# # Convolution: Requires input_shape, kernel_size, kernel_count.
# firstConvNetwerk.add(Convolution(input_shape=(359, 639, depth), kernel_size=3, kernel_count=2))

# # ReLU: No parameters required.
# firstConvNetwerk.add(ReLU())

# # MaxPool2D: Requires size and stride.
# firstConvNetwerk.add(MaxPool2D(size=2, stride=2))

# # Flatten: No parameters required.
# firstConvNetwerk.add(Flatten())

# # Dense: Requires input_size and output_size.
# firstConvNetwerk.add(Dense(input_size=113208, output_size=10))

# # Softmax: No parameters required.
# firstConvNetwerk.add(Softmax())

# firstConvNetwerk.train("archive/TUSimple/train_set/label_data_0313.json", 1, 64, 0.001)