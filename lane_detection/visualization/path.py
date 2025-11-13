from visualization.visualize_lanes import Lane_Visual
import torch
import cv2
import json
import numpy as np
from torchvision import transforms
from pytorch_architectures.resnet18 import ResNet18
from training_tools.maskedMSELoss import MaskedMSELoss


# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ResNet18(4, 48)
model_path = "models/mini_batch_gradient_descent/trial2_lower_batch_size/weights.pth"
ckpt = torch.load(model_path, map_location=device)

# Strict load to be sure shapes/keys match
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.to(device)
model.eval()
print("Model loaded successfully")


# Setup preprocessing (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


criterion = MaskedMSELoss()

def display_gps_path(image_path, direction):

    image = cv2.imread(image_path) # H, W, C
    if image is None:
        print(f"X Image not found: {image_path}")


    # Convert and preprocess image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)


    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)                         # (1,4,48)

    # Visualize Lanes
    current_lane_visual = Lane_Visual(outputs, (image.shape[1], image.shape[0]), image, switch_direction=direction)
    return current_lane_visual.get_image(markers=False, center_lane=True, left_lane=True, right_lane=True, lane_merge=True, visualize=True)


cv2.destroyAllWindows()
