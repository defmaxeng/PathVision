import os
import json
from PIL import Image

# Paths
json_file = "archive/TUSimple/train_set/label_data_0313.json"             # Path to your JSON file

image_root = "archive/TUSimple/train_set"               # Folder that contains 'clips/'
output_root = "images/256x144"                        # Output folder for resized images
output_json = f"{output_root}/label_data_0313_256x144.json"

# Make sure output folder exists
os.makedirs(output_root, exist_ok=True)

# with open(json_file, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         rel_path = data["raw_file"]            # e.g. "clips/0313-1/6040/20.jpg"
#         input_path = os.path.join(image_root, rel_path)
#         output_path = os.path.join(output_root, rel_path)

#         try:
#             # Load and resize image
#             img = Image.open(input_path).convert('RGB')
#             img = img.resize((256, 144))

#             # Make sure subdirectories exist in output
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)

#             # Save image
#             img.save(output_path)
#             print(f"Saved: {output_path}")
#         except Exception as e:
#             print(f"Failed: {input_path} — {e}")


x_scale = 1/5
y_scale = 1/5

with open(json_file, 'r') as infile, open(output_json, 'w') as outfile:
    for line in infile:
        data = json.loads(line)

        # Scale h_samples
        data['h_samples'] = [int(h * y_scale) for h in data['h_samples']]

        # Scale x values in each lane, skip -2
        scaled_lanes = []
        for lane in data['lanes']:
            scaled_lane = [int(x * x_scale) if x != -2 else -2 for x in lane]
            scaled_lanes.append(scaled_lane)
        data['lanes'] = scaled_lanes

        # Write the scaled data to output JSON
        json.dump(data, outfile)
        outfile.write('\n')