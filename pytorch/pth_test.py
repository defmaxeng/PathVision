import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
import numpy as np
from torchvision import transforms
from model import ConvNet

# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ConvNet(4, 48)
model_path = "saved_models/residual/residual_1/model-1/checkpoint.pth"
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
print("Model loaded successfully ✅")

# Setup preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Setup criterion
class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / mask.sum().clamp(min=1)

criterion = MaskedMSELoss()

# ========== Process JSON File Sequentially ==========
json_path = "images/256x144/label_data_0313_256x144.json"
resolution = "256x144"
width = 256

print("Starting sequential processing of JSON file...")
print("Controls: Press 'q' to quit, any other key to continue to next image")
print("="*60)

total_images = 0
processed_images = 0
total_loss = 0
total_correct = 0
total_valid_points = 0

with open(json_path, 'r') as f:
    for line_num, line in enumerate(f):
        total_images += 1
        
        try:
            # Parse JSON data
            data = json.loads(line)
            raw_file = data['raw_file']
            lanes = data['lanes']
            h_samples = data['h_samples']
            
            # Load image
            image_path = f"images/{resolution}/{raw_file}"
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Image not found: {image_path}")
                continue
                
            print(f"\n📸 Processing image {line_num + 1}: {raw_file}")
            
            # Convert and preprocess image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).unsqueeze(0).to(device)
            
            # Prepare ground truth
            ground_truth_lanes = np.array(lanes, dtype=np.float32)
            mask_np = (ground_truth_lanes != -2).astype(np.float32)
            ground_truth_processed = ground_truth_lanes.copy()
            ground_truth_processed[ground_truth_processed == -2] = 0.0
            
            # Run inference
            with torch.no_grad():
                prediction = model(image_tensor).squeeze(0).cpu().numpy()
            
            # Calculate loss
            gt_tensor = torch.tensor(ground_truth_processed).unsqueeze(0)
            mask_tensor = torch.tensor(mask_np).unsqueeze(0)
            pred_tensor = torch.tensor(prediction).unsqueeze(0)
            loss = criterion(pred_tensor, gt_tensor, mask_tensor).item()
            
            # Calculate accuracy
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
            
            # Create visualization
            display_image = image.copy()
            
            # Draw predicted (green) and ground truth (red)
            for line_idx, (pred_line, gt_line) in enumerate(zip(prediction, ground_truth_lanes)):
                for y, (px, gx) in enumerate(zip(pred_line, gt_line)):
                    if y >= len(h_samples):
                        break  # safety
                    yy = h_samples[y]

                    if gx != -2:  # Valid ground truth point
                        cv2.circle(display_image, (int(gx), int(yy)), 1, (0, 0, 255), -1)  # Red = GT

                    if px >= 0 and 0 <= px <= width:  # Valid prediction point
                        cv2.circle(display_image, (int(px), int(yy)), 1, (0, 255, 0), -1)  # Green = Pred
            
            # Add text overlay with results
            # cv2.putText(display_image, f"Image {line_num + 1}: {raw_file}", (10, 25), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(display_image, f"Loss: {loss:.2f}", (10, 50), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # cv2.putText(display_image, f"Accuracy: {accuracy:.1f}%", (10, 75), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # cv2.putText(display_image, f"Valid points: {int(valid_points)}", (10, 100), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # cv2.putText(display_image, "Red: GT, Green: Pred | Press 'q' to quit", (10, 125), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Calculate and display bias
            all_errors = []
            for lane_idx in range(len(ground_truth_lanes)):
                valid_mask = mask_np[lane_idx] > 0
                if valid_mask.sum() > 0:
                    errors = prediction[lane_idx][valid_mask] - ground_truth_lanes[lane_idx][valid_mask]
                    all_errors.extend(errors.tolist())
            
            if all_errors:
                mean_bias = np.mean(all_errors)
                cv2.putText(display_image, f"Mean bias: {mean_bias:.1f}px", (10, 145), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow("Lane Detection Sequential", display_image)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n🛑 User quit. Exiting...")
                break
                
        except Exception as e:
            print(f"❌ Error processing line {line_num + 1}: {str(e)}")
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