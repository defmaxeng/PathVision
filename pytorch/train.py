import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2
import json
import os

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

base_dir = "saved_models"
model_num = 0

def get_next_model_dir(base_dir, prefix="model-"):
    i = 1
    while True:
        folder_name = f"{prefix}{i}"
        full_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

            return full_path  # e.g., 'saved_models/model-3'
        i += 1




#############################################################
# Dataset
json_file_path = "images/256x144/label_data_0313_256x144.json"
resolution = "256x144"




################################################################
# Define the model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (B, 1, 28, 28) -> (B, 16, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                           # (B, 16, 28, 28) -> (B, 16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (B, 32, 14, 14)
        self.fc1 = nn.Linear(32 * 64 * 36, 4*48)                     # After 2x maxpool -> 7x7 feature maps

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = x.view(-1, 4, 48)
        return x





#################################################################
# error

def masked_mse_loss(pred, target, mask):
    """
    pred:   (B, L, H)  predicted x-positions
    target: (B, L, H)  ground-truth x-positions (dummy where mask==0)
    mask:   (B, L, H)  1 where valid label, 0 where missing (-2 in JSON)
    """
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, mask):
        return masked_mse_loss(pred, target, mask)

    

################################################################
# Training Setup
model = ConvNet()
criterion = MaskedMSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





model.to(device)




#################################################################
# training

def train(model, json_file_path, criterion, optimizer, epochs):
        
    # Create the next model folder
    model_dir = get_next_model_dir(base_dir)
    model_name = os.path.basename(model_dir)  # e.g. "model-3"
    model_num = int(model_name.split("-")[1])  # → 3



    
    # Image pre-processing (resize to model input shape, normalize, etc.)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),            # (H, W, C) → (C, H, W) + normalize to [0,1]
    ])
    # Read JSON lines
    with open(json_file_path, 'r') as file:
        lines = file.readlines()
    threshold = 5.0  # allow ±5 pixel tolerance
    print_every = 100

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        total = 0
        os.makedirs(f"saved_models/model-{model_num}/epoch-{epoch}")
        for idx, line in enumerate(lines):
            label_data = json.loads(line)
            lanes = label_data['lanes']
            h_samples = label_data['h_samples']
            raw_file = label_data['raw_file']

            lanes_tensor = torch.tensor(lanes, dtype=torch.float32)
            mask = (lanes_tensor != -2).float()
            lanes_tensor[lanes_tensor == -2] = 0.0

            image_path = f"images/{resolution}/{raw_file}"
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = preprocess(image).unsqueeze(0)

            # Move to device
            lanes_tensor = lanes_tensor.to(device)
            mask = mask.to(device)
            image_tensor = image_tensor.to(device)


            # Forward pass
            outputs = model(image_tensor)
            loss = criterion(outputs, lanes_tensor.unsqueeze(0), mask.unsqueeze(0))

            


            # Accuracy (per-point within threshold)
            with torch.no_grad():
                diffs = torch.abs(outputs - lanes_tensor.unsqueeze(0))
                correct = ((diffs < threshold) * mask.unsqueeze(0)).sum()
                total_masked = mask.sum()
                acc = 100 * correct / total_masked if total_masked > 0 else 0

            if (idx + 1) % print_every == 0:
                print(f"[Epoch {epoch}] Image {idx+1}: Loss = {loss.item():.4f}, Accuracy = {acc:.2f}%")
                print(f"correct = {correct}, total_masked = {total_masked}")
                torch.save(model.state_dict(), f'saved_models/model-{model_num}/epoch-{epoch}/image-{idx+1}.pth')


            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += 1

        print(f"Epoch {epoch} done: Avg Loss = {total_loss / total:.4f}")
        # Round and format loss
        formatted_loss = f"{total_loss / total:.2f}"

        # New folder name with loss
        new_model_dir = f"{model_dir}/epoch-{epoch}_avgLoss={formatted_loss}"

        # Rename the folder
        os.rename(f"{model_dir}/epoch-{epoch}", new_model_dir)
        print(f"📦 Renamed model folder to: {new_model_dir}")
        


if __name__ == "__main__":

    train(model, json_file_path, criterion, optimizer, epochs=50)




