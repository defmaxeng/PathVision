import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvNet
from torchvision import datasets, transforms
import cv2
import json
import os
import argparse


# Base directory is the folder that holds the model folders generated in this program
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--json_path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--config", type=str, default=None)

args = parser.parse_args()
if args.width is None or args.height is None:
    raise ValueError("Both --width and --height must be provided.")



print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

base_dir = args.base_dir
model_num = 0

def get_next_model_dir(base_dir, prefix="model-"):
    i = 1
    while True:
        folder_name = f"{prefix}{i}"
        # print("basedir: ", base_dir)
        # print("model_name: ", folder_name)
        full_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print("made new model folder: ", full_path)
            return full_path  # e.g., 'saved_models/model-3'
        i += 1




#############################################################
# Dataset
json_file_path = "images/256x144/label_data_0313_256x144.json"
resolution = "256x144"






#################################################################
# error

def masked_mse_loss(pred, target, mask):
    """
    pred:   (B, L, H)  predicted x-positions
    target: (B, L, H)  ground-truth x-positions (dummy where mask==0)
    mask:   (B, L, H)  1 where valid label, 0 where missing (-2 in JSON)
    """ 
    # print("Predicted shape: ", pred.size())
    # print("Target Shape: ", target.size())
    diff2 = (pred - target) ** 2 * mask
   
    return diff2.sum() / mask.sum().clamp(min=1)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, mask):
        return masked_mse_loss(pred, target, mask)

    

################################################################
# Training Setup
model = ConvNet(4, 48)
criterion = MaskedMSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





model.to(device)
import shutil

def reinitialize_model():
    print("reinitializing model...")
    new_model = ConvNet(4, 48).to(device)
    
    highest_model_index = 2
    while os.path.exists(f"{base_dir}/model-{highest_model_index}"):
        highest_model_index += 1
        print("newhighestmodel: ", highest_model_index)
    print(f"deleting: {base_dir}/model-{highest_model_index-1}")
    shutil.rmtree(f"{base_dir}/model-{highest_model_index-1}")

    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=args.lr)
    train(new_model, json_file_path, criterion, new_optimizer, epochs=args.epochs)



#################################################################
# training
def train(model, json_file_path, criterion, optimizer, epochs):
    nan_detected = False  # Initialize the variable
    
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
    print_every = 100  # how often to log intermediate loss
    loss_file_path = os.path.join(model_dir, "loss_dump.txt")

    with open(loss_file_path, "w") as f:
        for epoch in range(epochs):
            total_loss = 0
            total = 0

            for idx, line in enumerate(lines):
                label_data = json.loads(line)
                lanes = label_data['lanes']
                raw_file = label_data['raw_file']

                lanes_tensor = torch.tensor(lanes, dtype=torch.float32)
                mask = (lanes_tensor != -2).float()
                lanes_tensor[lanes_tensor == -2] = 0.0

                image_path = f"images/{resolution}/{raw_file}"
                image = cv2.imread(image_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                lanes_tensor = lanes_tensor.to(device)
                mask = mask.to(device)
                # print("image tensor shape: ", image_tensor.size())
                outputs = model(image_tensor)
                loss = criterion(outputs, lanes_tensor.unsqueeze(0), mask.unsqueeze(0))

                # Check for NaN loss
                if torch.isnan(loss):
                    nan_detected = True
                    print(f"NaN detected at epoch {epoch}, image {idx+1}")
                    f.write(f"NaN detected at epoch {epoch}, image {idx+1}\n")
                    f.close()
                    reinitialize_model()

                    return  # Break out of the inner loop
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += 1

                if (idx + 1) % print_every == 0:
                    with torch.no_grad():
                        diffs = torch.abs(outputs - lanes_tensor.unsqueeze(0))
                        correct = ((diffs < threshold) * mask.unsqueeze(0)).sum()
                        total_masked = mask.sum()
                        acc = 100 * correct / total_masked if total_masked > 0 else 0
                    log_line = f"[Epoch {epoch}] Image {idx+1}: Loss = {loss.item():.4f}, Accuracy = {acc:.2f}%"
                    print(log_line)

            

            if total > 0:
                avg_loss = total_loss / total
                epoch_line = f"Epoch {epoch} done: Avg Loss = {avg_loss:.4f}"
                print(epoch_line)
                f.write(epoch_line + "\n")
            else:
                epoch_line = f"Epoch {epoch} skipped: no valid images"
                print(epoch_line)
                f.write(epoch_line + "\n")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, f'{model_dir}/checkpoint.pth')
        


if __name__ == "__main__":

    train(model, json_file_path, criterion, optimizer, epochs=args.epochs)




