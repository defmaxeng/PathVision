import torch
from pytorch_architectures.resnet18 import ResNet18
from training_tools.maskedMSELoss import MaskedMSELoss
from models.mini_batch_gradient_descent.trial3_bs8_1280x720.train import mbgd

# Get base directory
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default=None)
args = parser.parse_args()
base_dir = args.base_dir # got the base directory

# Define Variables
model = ResNet18(4, 48)
learning_rate = 0.001
epochs = 10
resolution = "1280/720"
criterion = MaskedMSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
print_every = 50
batch_size = 8

# Files
json_file_path = "datasets/archive/TUSimple/train_set/label_data_0313.json"

mbgd(model, json_file_path, criterion, optimizer, epochs, resolution, base_dir, batch_size)
