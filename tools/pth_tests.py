import subprocess

subprocess.run([
    "python", "pytorch/overlay_predictions.py",
    "--model_path", "saved_models/residual/residual_1/model-3/checkpoint.pth",
    "--json_path", "images/256x144/label_data_0313_256x144.json",
    "--width", "256",
    "--height", "144",
    "--num_images", "10"
])\

