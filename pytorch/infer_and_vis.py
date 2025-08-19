import os, json, cv2, numpy as np, torch
import torch.nn as nn
from torchvision import transforms

# ====== CONFIG you can tweak ======
MODEL_W, MODEL_H = 256, 144        # your training/inference size
THRESHOLD_PX = 5.0                 # accuracy threshold (±px)
SHOW_WINDOWS = False               # set True to cv2.imshow, False to save files
SAVE_DIR = "viz_out"               # folder to save visualizations if SHOW_WINDOWS=False
IMAGE_ROOT = "images/256x144"      # where raw_file lives; change if needed

# ====== same preprocessing as training (ToTensor only) ======
PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # If you trained with Normalize, add the SAME stats here:
    # transforms.Normalize(mean=[0.485,0.456,0.406],
    #                      std=[0.229,0.224,0.225]),
])

# ====== masked MSE ======
class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        # pred/target/mask: (B, 4, 48) float32
        diff2 = (pred - target) ** 2 * mask
        denom = mask.sum().clamp(min=1.0)
        return diff2.sum() / denom

# ====== drawing util (handles scaling & NaNs) ======
def draw_lanes_on_image(image_bgr, prediction_4x48, h_samples, mask_4x48=None,
                        model_w=256, model_h=144, color=(0,255,0), thickness=2):
    """
    image_bgr:   BGR uint8 image
    prediction:  (4,48) x-coordinates in model pixel space (width=model_w)
    h_samples:   list/array of 48 y-coordinates (often from dataset space)
    mask_4x48:   (4,48) float mask (1 valid, 0 invalid); if None, draws all
    """
    # 1) Ensure canvas is exactly model size for consistent coords
    H0, W0 = image_bgr.shape[:2]
    if (W0, H0) != (model_w, model_h):
        image_bgr = cv2.resize(image_bgr, (model_w, model_h), interpolation=cv2.INTER_AREA)
    H, W = image_bgr.shape[:2]

    # 2) h_samples -> numpy, scale to canvas height if it looks taller (e.g., 720→144)
    h = np.array(h_samples, dtype=np.float32)
    max_h = float(h.max()) if h.size else 0.0
    if max_h > H + 1:
        h = (h * (H / max_h)).astype(np.int32)
    else:
        h = h.astype(np.int32)

    # 3) scale predicted x from model space to canvas (usually scale_x=1)
    pred = np.asarray(prediction_4x48, dtype=np.float32) * (W / float(model_w))

    # default mask: draw all
    if mask_4x48 is None:
        mask_4x48 = np.ones_like(pred, dtype=np.float32)

    out = image_bgr.copy()
    for lane_idx, lane_xs in enumerate(pred):
        # safety: replace NaN/Inf and clip to visible-ish range
        lane_xs = np.nan_to_num(lane_xs, nan=-1e9, posinf=1e9, neginf=-1e9)
        lane_xs = np.clip(lane_xs, -1, W)

        for i in range(1, len(h)):
            if not (mask_4x48[lane_idx, i-1] > 0.5 and mask_4x48[lane_idx, i] > 0.5):
                continue
            x1, y1 = int(lane_xs[i-1]), int(h[i-1])
            x2, y2 = int(lane_xs[i]),   int(h[i])
            if 0 <= x1 < W and 0 <= x2 < W and 0 <= y1 < H and 0 <= y2 < H:
                cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out

# ====== main runner ======
def run_inference_and_visualize(model, ckpt_path, json_path,
                                image_root=IMAGE_ROOT, save_dir=SAVE_DIR,
                                device=None, max_images=None):
    os.makedirs(save_dir, exist_ok=True)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # ---- load weights safely ----
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    criterion = MaskedMSELoss()

    # ---- read jsonl ----
    with open(json_path, "r") as f:
        lines = f.readlines()

    total_loss, total_acc, count = 0.0, 0.0, 0
    for idx, line in enumerate(lines):
        if max_images is not None and idx >= max_images:
            break

        d = json.loads(line)
        raw_file   = d["raw_file"]
        h_samples  = d["h_samples"]                 # list len 48
        lanes_np   = np.array(d["lanes"], dtype=np.float32)   # (4,48)

        # mask & clean GT
        mask_np = (lanes_np != -2).astype(np.float32)         # (4,48)
        lanes_np[lanes_np == -2] = 0.0

        # load image
        img_path = os.path.join(image_root, raw_file)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"⚠️  cannot read image: {img_path}")
            continue

        # ensure drawing canvas is model size (avoids “squish”)
        if (image_bgr.shape[1], image_bgr.shape[0]) != (MODEL_W, MODEL_H):
            image_bgr = cv2.resize(image_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

        # model input is RGB tensor
        image_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor= PREPROCESS(image_rgb).unsqueeze(0).to(device, dtype=torch.float32)  # (1,3,144,256)

        # ---- inference once ----
        with torch.no_grad():
            outputs = model(image_tensor)            # (1,4,48)
        pred_np = outputs.squeeze(0).detach().cpu().numpy()    # (4,48)

        # ---- loss / accuracy (masked) ----
        y = torch.from_numpy(lanes_np).unsqueeze(0).to(device, dtype=torch.float32)
        m = torch.from_numpy(mask_np).unsqueeze(0).to(device, dtype=torch.float32)

        loss = criterion(outputs.to(torch.float32), y, m).item()
        with torch.no_grad():
            diffs = (outputs - y).abs()
            correct = ((diffs < THRESHOLD_PX) * m).sum().item()
            total_valid = m.sum().item()
            acc = 100.0 * correct / total_valid if total_valid > 0 else 0.0

        total_loss += loss
        total_acc  += acc
        count += 1

        # ---- visualize ----
        vis = draw_lanes_on_image(
            image_bgr=image_bgr,                # BGR canvas
            prediction_4x48=pred_np,            # (4,48) x's
            h_samples=h_samples,                # 48 y's
            mask_4x48=mask_np,                  # (4,48)
            model_w=MODEL_W, model_h=MODEL_H,
            color=(0,255,0), thickness=2
        )

        # show or save
        if SHOW_WINDOWS:
            cv2.imshow(f"pred [{idx}] loss={loss:.3f} acc={acc:.2f}%", vis)
            key = cv2.waitKey(0)
            if key == 27:  # ESC to break
                break
        else:
            out_path = os.path.join(save_dir, os.path.basename(raw_file).replace("/", "_"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, vis)

        # optional quick logs
        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}] avg_loss={total_loss/count:.4f}  avg_acc={total_acc/count:.2f}%")

    if count > 0:
        print(f"✅ Done. Samples={count} | AvgLoss={total_loss/count:.4f} | AvgAcc±{THRESHOLD_PX}px={total_acc/count:.2f}%")
    else:
        print("⚠️ No valid samples processed.")

from model import ConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(out_cols=4, out_rows=48)

ckpt_path = "saved_models/residual/residual_1/model-3/checkpoint.pth"
json_path = "images/256x144/label_data_0313_256x144.json"

run_inference_and_visualize(
    model=model,
    ckpt_path=ckpt_path,
    json_path=json_path,
    image_root=IMAGE_ROOT,     # or "images/<your_resolution>"
    save_dir=SAVE_DIR,         # where to drop PNGs if SHOW_WINDOWS=False
    device=device,
    max_images=100             # or None for all
)