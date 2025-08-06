import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
import os
import argparse
import numpy as np
from torchvision import transforms

# ────────────────────────────────────────────────
# 🧠 Rebuilt ConvNet model
class ConvNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(int(32 * width * height / 16), 4 * 48)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, 4, 48)
        return x

# ────────────────────────────────────────────────
# 🎛️ CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--json_path", type=str, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--output_dir", type=str, default="predictions_overlay")
parser.add_argument("--num_images", type=int, default=10)
args = parser.parse_args()

# ────────────────────────────────────────────────
# 🖥️ Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(args.width, args.height).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

os.makedirs(args.output_dir, exist_ok=True)
resolution = f"{args.width}x{args.height}"

# ────────────────────────────────────────────────
# 📸 Prediction + Overlay
with open(args.json_path, 'r') as f:
    lines = f.readlines()
previous_tensor = None
for idx, line in enumerate(lines[:args.num_images]):
    data = json.loads(line)
    raw_file = data['raw_file']
    lanes = data['lanes']

    # Load image
    img_path = f"images/{resolution}/{raw_file}"
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping missing image: {img_path}")
        continue
    print("Predicting for:", img_path)

    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_img = preprocess(image_rgb).unsqueeze(0).to(device)
    print("Image sum:", tensor_img.sum().item())

    # if previous_tensor == tensor_img:
    #     print("all tensors are the same :(")
    # previous_tensor = tensor_img
    output = None  # wipe previous prediction
    with torch.no_grad():
        output = model(tensor_img)[0].cpu().numpy()  # (4, 48)

    h_samples = data["h_samples"]  # The true y-values for each x-point

    # Draw predicted (green) and ground truth (red)
    for line_idx, (pred_line, gt_line) in enumerate(zip(output, lanes)):
        # print(f"Image {idx}: mean pixel = {tensor_img.mean().item():.4f}, prediction mean = {output.mean():.2f}")


        for y, (px, gx) in enumerate(zip(pred_line, gt_line)):
            if y >= len(h_samples):
                break  # safety
            yy = h_samples[y]

            if gx != -2:
                cv2.circle(original, (int(gx), int(yy)), 2, (0, 0, 255), -1)  # Red = GT

            if px >= 0 and 0 <= px <= args.width:
                cv2.circle(original, (int(px), int(yy)), 2, (0, 255, 0), -1)  # Green = Pred

    save_path = os.path.join(args.output_dir, f"overlay_{idx}.png")
    cv2.imwrite(save_path, original)
    print(f"✅ Saved overlay: {save_path}")
