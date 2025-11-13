import numpy as np
import torch.nn as nn


class MaskedMSECubicBezierLoss(nn.Module):
    def __init__(self):
        super().__init__()



    def eval_cubic_bezier(outputs, TULoss, num_samples=100):
        """
        Outputs: [B, 5, 9]
        - first 4 elements in each row are x vals (in pixel values)
        - next 4 elements in each row are y vals (in pixel values)
        - last val is the confidence val (normalized [0, 1])

        TU Loss shape: [b, 4, 48]
         - 
        """

        t = np.linspace(0.0, 1.0, num_samples).reshape(-1, 1)
        B0 = (1 - t) ** 3
        B1 = 3 * (1 - t) ** 2 * t
        B2 = 3 * (1 - t) * (t ** 2)
        B3 = t ** 3
        # shape: (num_samples, 1) @ (1, 2) broadcasts to (num_samples, 2)
        pts = B0 * P[0] + B1 * P[1] + B2 * P[2] + B3 * P[3]


        
        return # The sum of all of the differences divided by t