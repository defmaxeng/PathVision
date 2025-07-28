from train import ConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2
import json

model = ConvNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  # or 'cuda'
model.eval()  # 🔒 puts model in inference mode


image_path = "images/256x144/clips/0313-1/6040/20.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# If using grayscale:
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0

# If using RGB:
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 144, 256)






with torch.no_grad():
    prediction = model(image_tensor)  # shape: (1, 4, 48) or whatever
    prediction = prediction.squeeze(0).numpy()  # shape: (4, 48)






import numpy as np

with open("images/256x144/label_data_0313_256x144.json", 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['raw_file'] == "clips/0313-1/6040/20.jpg":
            h_samples = data['h_samples']
            break
        
for lane in prediction:
    for i in range(1, len(h_samples)):
        x1, y1 = int(lane[i-1]), int(h_samples[i-1])
        x2, y2 = int(lane[i]), int(h_samples[i])
        if x1 >= 0 and x2 >= 0:  # skip invalid
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
