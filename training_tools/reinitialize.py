from pytorch_architectures.resnet18 import ResNet18
import shutil
import torch
from training_tools.train import train

def reinitialize_model(model_folder, json_file_path, criterion):
    # print("Loss too high -- reinitializing model with new weights")
    # new_model = ResNet18(4, 48).to(device)
    # shutil.rmtree(model_folder)
    
    # new_optimizer = torch.optim.SGD(new_model.parameters(), lr=args.lr)
    # train(new_model, json_file_path, criterion, new_optimizer, epochs=args.epochs)

