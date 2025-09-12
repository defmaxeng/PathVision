import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / mask.sum().clamp(min=1)