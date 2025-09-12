import torch
import os
from pytorch_architectures.resnet18 import ResNet18
from training_tools.maskedMSELoss import MaskedMSELoss
from training_tools.train import train

# Get base directory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default=None)
args = parser.parse_args()
base_dir = args.base_dir # got the base directory

# Define Variables
model = ResNet18(4, 48)
learning_rate = 0.1
epochs = 10
resolution = "256x144"
criterion = MaskedMSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
print_every = 50

# Files
json_file_path = "images/256x144/label_data_0313_256x144.json"

train(model, json_file_path, criterion, optimizer, epochs, resolution, base_dir, print_every)
