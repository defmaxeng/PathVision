import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ConvNet(nn.Module):
    def __init__(self, out_cols=4, out_rows=48, pretrained=True):
        """
        out_cols: numba of lane channels (4)
        out_rows: numba of samples per lane (48)
        """
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)  # first conv already 3->64
        in_feats = self.backbone.fc.in_features     # 512
        # Replace the classifier head to match 4*48 outputs
        self.backbone.fc = nn.Linear(in_feats, out_cols * out_rows)
        self.out_cols = out_cols
        self.out_rows = out_rows
        

    def forward(self, x):
        
        x = self.backbone(x)                        # (B, out_cols*out_rows)
        x = x.view(x.size(0), self.out_cols, self.out_rows)  # (B, 4, 48)
        
        return x