import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, out_cols=4, out_rows=48, pretrained=True):
        super().__init__()

        # Network Output Dimensions -> 4 lanes each with 48 points 
        self.out_cols = out_cols
        self.out_rows = out_rows

        # Initialize a resnet18 backbone using pretrained weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights) 

        # Resnet18 Outputs a 512 neuron long column of features
        res_output_features = self.backbone.fc.in_features     
        
        # Replaces Resnet's last fc layer with our own fc layer that has the amount of outputs we want (192) because 4 * 48
        self.backbone.fc = nn.Linear(res_output_features, out_cols * out_rows)



    def forward(self, x):
        # input has shape: (B, 3, H, W)

        # Passes through the 18 layer Resnet Backbone
        x = self.backbone(x)                        # (B, out_cols*out_rows)
        x = x.view(x.size(0), self.out_cols, self.out_rows)  # (B, out_cols, out_rows)
        
        return x