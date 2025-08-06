import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
import numpy as np
from torchvision import transforms
import argparse


# Base directory is the folder that holds the model folders generated in this program
parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", type=str, default=None)
parser.add_argument("--imagepath", type=str, default=None)
parser.add_argument("--jsonpath", type=str, default=None)
parser.add_argument("--imgrelpath", type=str, default=None)
args = parser.parse_args()


# ========== Model Definition ==========
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 36, 4 * 48)  # output is (1, 4, 48)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, 4, 48)
        return x


# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ConvNet()
model_path = args.modelpath  # change to match your model folder
# "saved_models/learning_rates_test_0.5-20/model-4/weights.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
print("Model loaded successfully ✅")


# ========== Load Test Image ==========
image_path = args.imagepath  # change if needed
# "images/256x144/clips/0313-2/7560/20.jpg"
image = cv2.imread(image_path)
print("recieved image: ", image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 144, 256)

# ========== Load h_samples ==========
json_path = args.jsonpath
# "images/256x144/label_data_0313_256x144.json"
with open(json_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data["raw_file"] == args.imgrelpath:
            h_samples = data["h_samples"]
            break
    else:
        raise ValueError("Could not find h_samples for test image.")


# ========== Run Inference ==========
with torch.no_grad():
    prediction = model(image_tensor).squeeze(0).cpu().numpy()  # shape: (4, 48)

# Setup criterion
class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / mask.sum().clamp(min=1)

criterion = MaskedMSELoss()

ground_truth_lanes = np.array(data['lanes'], dtype=np.float32)
mask_np = (ground_truth_lanes != -2).astype(np.float32)
ground_truth_lanes[ground_truth_lanes == -2] = 0.0

# Convert to torch for loss
gt_tensor = torch.tensor(ground_truth_lanes).unsqueeze(0)
mask_tensor = torch.tensor(mask_np).unsqueeze(0)
pred_tensor = torch.tensor(prediction).unsqueeze(0)

# Compute loss
loss = criterion(pred_tensor, gt_tensor, mask_tensor).item()

# Accuracy
threshold = 5.0
abs_diff = np.abs(prediction - ground_truth_lanes)
correct = ((abs_diff < threshold) * mask_np).sum()
total_valid = mask_np.sum()
accuracy = 100 * correct / total_valid if total_valid > 0 else 0.0

print(f"📉 Loss: {loss:.2f}")
print(f"✅ Accuracy within ±{threshold} pixels: {accuracy:.2f}%")

# ========== Draw Prediction ==========
output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for lane in prediction:
    for i in range(1, len(h_samples)):
        x1, y1 = int(lane[i - 1]), int(h_samples[i - 1])
        x2, y2 = int(lane[i]), int(h_samples[i])
        if x1 >= 0 and x2 >= 0:
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Predicted Lanes", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
