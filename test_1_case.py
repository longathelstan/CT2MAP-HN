import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from monai.networks.nets import BasicUNet
import os

# 1. Load config and model
config_path = "configs/baseline_p40.yaml"
ckpt_path = "checkpoints/baseline_p40/best_model.pth"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# The custom model from your project
from src.models.baseline_unet import BaselineUNet

# Init model
model = BaselineUNet(
    in_channels=1,
    features=(32, 32, 64, 128, 256, 32),
    dropout=0.0,
    use_uncertainty=False,
    norm="instance"
)
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.cuda()
model.eval()

# 2. Load 1 sample data (processed numpy array)
npy_path = "data/processed/CHUM-001_ct.npy"
print(f"Loading {npy_path}...")
img = np.load(npy_path) # shape: (D, H, W)
# Add batch and channel dimension
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()

print(f"Input shape: {img_tensor.shape}")

# 3. Inference
with torch.no_grad():
    with torch.cuda.amp.autocast():
        output = model(img_tensor)

heatmap = output["heatmap"].squeeze().cpu().numpy()
print(f"Output Heatmap shape: {heatmap.shape}, min={heatmap.min():.4f}, max={heatmap.max():.4f}")

# 4. Plot middle slice
D = img.shape[0]
mid_d = D // 2

plt.figure(figsize=(15, 5))

# Original CT slice
plt.subplot(1, 3, 1)
plt.imshow(img[mid_d], cmap="bone")
plt.title(f"CT Slice (Z={mid_d})")
plt.axis("off")

# Predicted Heatmap
plt.subplot(1, 3, 2)
plt.imshow(heatmap[mid_d], cmap="jet", vmin=0, vmax=1)
plt.title("Predicted Lesion Heatmap")
plt.axis("off")

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(img[mid_d], cmap="bone")
plt.imshow(heatmap[mid_d], cmap="jet", alpha=0.4, vmin=0, vmax=1)
plt.title("Overlay")
plt.axis("off")

out_png = "/home/long/.gemini/antigravity-ide/brain/6343d00c-04af-4035-b197-b6389c3528ed/inference_demo.png"
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"Plot saved to {out_png}")
