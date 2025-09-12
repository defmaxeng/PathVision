import json
import cv2
import torch
from torchvision import transforms

def train(model, json_file_path, criterion, optimizer, epochs, resolution, base_dir, print_every=50):
    # Checks that GPU is functional
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # put the model in the gpu

    # Put the model in training mode
    model.train()

    # Define preprocessing to turn image into the tensor
    preprocess = transforms.Compose([
            transforms.ToPILImage(), # (H, W, RGB)
            transforms.ToTensor(), # RGB, H, W, pixel values normalized [0, 1]
    ])                

    # Read the lines of the json file
    with open(json_file_path, 'r') as f:
        lines = f.readlines()

    # Start Writing
    with open(f"{base_dir}/loss.txt", "w") as f:
        for epoch_num in range(epochs):
            epoch_loss = 0.0  # keep track of epoch_loss
            num_images = 0
            for index, json_line in enumerate(lines):
                num_images += 1
                # 1. Load in the raw image and lanes
                loaded_line = json.loads(json_line)
                raw_image_path = f"images/{resolution}/{loaded_line['raw_file']}"
                raw_image = cv2.imread(raw_image_path) # in BGR rn
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                lanes = loaded_line["lanes"]

                # 2. Convert to tensor: raw_img, label lanes, mask
                image_tensor = preprocess(raw_image).unsqueeze(0).to(device) # Preprocesses, 1, RGB, h, w -> Pytorch expects the number in front -> batch size
                lanes_tensor = torch.tensor(lanes, dtype=torch.float32)
                mask_tensor = (lanes_tensor != -2).float()
                lanes_tensor[lanes_tensor == -2] = 0.0

                # 3. Move image, label lanes, and mask to the gpu
                image_tensor.to(device)
                lanes_tensor = lanes_tensor.to(device)
                mask_tensor = mask_tensor.to(device)
                
                # 4. Run the network -> 5 steps
                optimizer.zero_grad(set_to_none=True)                                   # 1. Clear the Gradient
                outputs = model(image_tensor)                                           # 2. For-Prop
                loss = criterion(outputs, lanes_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)) # 3. Calculate Loss
                loss.backward()                                                         # 4. Back-Prop
                optimizer.step()                                                        # 5. Gradient Descent

                # 5. Print out average loss regularly
                epoch_loss += loss.item() # loss.item() is the loss per image, add to total epoch loss
                if (index + 1) % print_every == 0:
                    with torch.no_grad():
                        avg_loss = epoch_loss / num_images
                        epoch_loss = 0
                        num_images = 1
                        print(f"Epoch [{epoch_num+1}/{epochs}], Image [{index+1}], Avg Loss: {avg_loss:.4f}")

            # Save output
            f.write(f"Epoch [{epoch_num+1}/{epochs}], Image [{index+1}], Avg Loss: {avg_loss:.4f}\n")

            # Save Network, will keep overwriting till the end
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
            }, f"{base_dir}/weights.pth")
                


