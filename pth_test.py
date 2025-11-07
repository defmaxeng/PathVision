import torch
import cv2
import json
import numpy as np
from torchvision import transforms
from pytorch_architectures.resnet18 import ResNet18
from training_tools.maskedMSELoss import MaskedMSELoss

# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ResNet18(4, 48)
model_path = "models/mini_batch_gradient_descent/trial2_lower_batch_size/weights.pth"
ckpt = torch.load(model_path, map_location=device)

# Strict load to be sure shapes/keys match
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.to(device)
model.eval()
print("Model loaded successfully")


# Setup preprocessing (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


criterion = MaskedMSELoss()

# ========== Process JSON File Sequentially ==========
json_path = "images/256x144/label_data_0313_256x144.json"
resolution = 256, 144
base_dir = "images/256x144"



print("Starting sequential processing of JSON file...")
print("Controls: Press 'q' to quit, any other key to continue to next image")
print("="*60)

total_images = 0
processed_images = 0
total_loss = 0
total_correct = 0
total_valid_points = 0

with open(json_path, 'r', encoding="utf-8") as f:
    for line_num, line in enumerate(f):
        total_images += 1
        try:
            # Parse JSON data
            data = json.loads(line)
            raw_file = data['raw_file']
            lanes = data['lanes']
            h_samples = data['h_samples']

            # Load image
            image_path = f"{base_dir}/{raw_file}"
            image = cv2.imread("images/256x144/clips/0313-1/6040/20.jpg")
            if image is None:
                print(f"X Image not found: {image_path}")
                continue

            # (Optional) ensure draw/eval canvas is 256x144 to match labels
            if (image.shape[1], image.shape[0]) != resolution:
                image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)

            print(f"\nProcessing image {line_num + 1}: {raw_file}")

            # Convert and preprocess image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0).to(device)

            # Prepare ground truth
            ground_truth_lanes = np.array(lanes, dtype=np.float32)    # (4,48)
            mask_np = (ground_truth_lanes != -2).astype(np.float32)   # (4,48)
            ground_truth_processed = ground_truth_lanes.copy()
            ground_truth_processed[ground_truth_processed == -2] = 0.0

            # Run inference
            with torch.no_grad():
                outputs = model(image_tensor)                         # (1,4,48)
                prediction = outputs.squeeze(0).cpu().numpy()         # (4,48)

            # Calculate loss (same masked MSE as train)
            gt_tensor   = torch.tensor(ground_truth_processed).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask_np).unsqueeze(0).to(device)
            loss = criterion(outputs, gt_tensor, mask_tensor).item()

            # Accuracy
            threshold = 5.0
            abs_diff = np.abs(prediction - ground_truth_processed)
            correct = ((abs_diff < threshold) * mask_np).sum()
            valid_points = mask_np.sum()
            accuracy = 100 * correct / valid_points if valid_points > 0 else 0.0

            # Update totals
            processed_images += 1
            total_loss += loss
            total_correct += correct
            total_valid_points += valid_points

            # Print results
            print(f"Loss: {loss:.2f}, Accuracy: {accuracy:.1f}%, Valid points: {int(valid_points)}")

            
            # Visualization (points)
            display_image = image.copy()
            for line_idx, (pred_line, gt_line) in enumerate(zip(prediction, ground_truth_lanes)):
                for y, (px, gx) in enumerate(zip(pred_line, gt_line)):
                    if y >= len(h_samples): break
                    yy = h_samples[y]
                    if gx != -2:
                        cv2.circle(display_image, (int(gx), int(yy)), 1, (0, 0, 255), -1)  # GT red
                    if px >= 0 and 0 <= px <= resolution[0]:
                        cv2.circle(display_image, (int(px), int(yy)), 1, (0, 255, 0), -1)  # Pred green

            # Mean bias (debug)
            all_errors = []
            for lane_idx in range(len(ground_truth_lanes)):
                valid_mask = mask_np[lane_idx] > 0
                if valid_mask.sum() > 0:
                    errors = prediction[lane_idx][valid_mask] - ground_truth_lanes[lane_idx][valid_mask]
                    all_errors.extend(errors.tolist())
            if all_errors:
                mean_bias = np.mean(all_errors)
                cv2.putText(display_image, f"Mean bias: {mean_bias:.1f}px", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Lane Detection Sequential", display_image)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\nUser quit. Exiting...")
                break

        except Exception as e:
            print(f"Error processing line {line_num + 1}: {str(e)}")
            continue

cv2.destroyAllWindows()

# ========== Final Summary ==========
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Total images in JSON: {total_images}")
print(f"Successfully processed: {processed_images}")
print(f"Failed to process: {total_images - processed_images}")

if processed_images > 0:
    avg_loss = total_loss / processed_images
    overall_accuracy = 100 * total_correct / total_valid_points if total_valid_points > 0 else 0.0
    print(f"\nAverage loss: {avg_loss:.2f}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Total valid points: {int(total_valid_points)}")
    print(f"Total correct points: {int(total_correct)}")
else:
    print("\nNo images were successfully processed.")
