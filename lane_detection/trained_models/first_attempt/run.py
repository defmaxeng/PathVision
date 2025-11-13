import torch
from pytorch_architectures.resnet18 import ResNet18
from training_tools.maskedMSELoss import MaskedMSELoss
from training_tools.train import train


# Define Variables
base_dir = "models/first_attempt"
model = ResNet18(4, 48)
learning_rate = 0.01
epochs = 50
resolution = "256x144"
criterion = MaskedMSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
print_every = 50

# Files
json_file_path = "images/256x144/label_data_0313_256x144.json"

train(model, json_file_path, criterion, optimizer, epochs, resolution, base_dir, print_every)
