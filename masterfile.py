import subprocess
import numpy as np
import os
import shutil

# Fixed arguments
base_dir = "saved_models/residual/residual_1"
os.makedirs(base_dir, exist_ok=True)
shutil.copy(os.path.abspath(__file__), os.path.abspath(base_dir))
shutil.copy(os.path.abspath("pytorch/model.py"), os.path.abspath(base_dir))
width = 256
height = 144
json_path = "images/256x144/label_data_0313_256x144.json"  # even if ignored, you gotta pass something since it's in args
epochs = 15
config = "configA"

# Learning rates to test
lrs = np.linspace(0.1, 5, num=6)  # 10 values from 0.005 to 0.1

for i, lr in enumerate(lrs, 1):
    print(f"▶️ Starting training #{i} with learning rate {lr:.5f}")

    cmd = [
        "python", "pytorch/train.py",
        "--base_dir", base_dir,
        "--width", str(width),
        "--height", str(height),
        "--json_path", json_path,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--config", config
    ]

    subprocess.run(cmd)