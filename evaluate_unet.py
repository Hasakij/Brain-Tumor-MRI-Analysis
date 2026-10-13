import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from monai.metrics import compute_hausdorff_distance
import numpy as np
import matplotlib.pyplot as plt

from dataset_unet import LGGDataset
from unet_model import Unet

# Configuration & Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/adam/.cache/kagglehub/datasets/mri_unet/lgg-mri-segmentation/kaggle_3m")
MODEL_PATH = "best_unet.pth"
THRESHOLD = 0.5  # Probability cutoff for binary tumor prediction


# Data Loading & Validation Split
all_images = [p for p in DATA_DIR.rglob("*.tif") if "_mask" not in p.stem]
_, val_paths = train_test_split(all_images, test_size=0.2, random_state=42)

val_ds = LGGDataset(image_paths=val_paths, transform=None)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Model Initialization
model = Unet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model weights from {MODEL_PATH}")

# Metric accumulators
total_tp = torch.tensor(0.0, device=device)
total_fp = torch.tensor(0.0, device=device)
total_fn = torch.tensor(0.0, device=device)
total_tn = torch.tensor(0.0, device=device)

# Dedicated metrics for slices containing ground-truth tumor
tumor_only_dices = []
hd95_values = []

# Storage for visual showcase
sample_images, sample_masks, sample_preds = [], [], []

print("Running validation inference...")

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Mixed precision inference
        with torch.amp.autocast('cuda'):
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).float()

        # Flatten predictions and targets to compute global confusion matrix
        p_flat = preds.view(-1)
        t_flat = masks.view(-1)

        total_tp += (p_flat * t_flat).sum()
        total_fp += (p_flat * (1.0 - t_flat)).sum()
        total_fn += ((1.0 - p_flat) * t_flat).sum()
        total_tn += ((1.0 - p_flat) * (1.0 - t_flat)).sum()

        # Per-sample evaluation for tumor slices (MONAI HD95 & Tumor-only Dice)
        for i in range(images.size(0)):
            if masks[i].sum() > 0:
                # Dice score for this specific slice
                intersection = (preds[i] * masks[i]).sum()
                dice = (2.0 * intersection) / (preds[i].sum() + masks[i].sum() + 1e-6)
                tumor_only_dices.append(dice.item())

                # MONAI HD95: requires (B, C, H, W) tensor input
                if preds[i].sum() > 0:
                    hd = compute_hausdorff_distance(
                        preds[i].unsqueeze(0),
                        masks[i].unsqueeze(0),
                        percentile=95
                    )
                    hd95_values.append(hd.item())

                # Collect a small subset of tumor slices for visual plot
                if len(sample_images) < 4:
                    # Convert CHW torch tensor to HWC numpy array for matplotlib display
                    sample_images.append(images[i].detach().cpu().permute(1, 2, 0).numpy())
                    sample_masks.append(masks[i].detach().cpu().squeeze().numpy())
                    sample_preds.append(preds[i].detach().cpu().squeeze().numpy())

# Metrics
eps = 1e-6
dice_all = ((2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)).item() * 100
iou_all = ((total_tp + eps) / (total_tp + total_fp + total_fn + eps)).item() * 100
precision = ((total_tp + eps) / (total_tp + total_fp + eps)).item() * 100
recall = ((total_tp + eps) / (total_tp + total_fn + eps)).item() * 100
specificity = ((total_tn + eps) / (total_tn + total_fp + eps)).item() * 100

dice_tumor_only = (np.mean(tumor_only_dices) if tumor_only_dices else 0.0) * 100
avg_hd95 = np.mean(hd95_values) if hd95_values else 0.0

print("U-NET EVALUATION")
print(f"Dice Score (All slices): {dice_all:.2f}%")
print(f"Dice Score (Tumor slices only): {dice_tumor_only:.2f}%")
print(f"IoU / Jaccard Index: {iou_all:.2f}%")
print(f"Recall / Sensitivity: {recall:.2f}%")
print(f"Precision / PPV: {precision:.2f}%")
print(f"Specificity: {specificity:.2f}%")
print(f"MONAI HD95 (Boundary error): {avg_hd95:.2f} px")

Path("results").mkdir(exist_ok=True)

# Plot 1: metrics bar
metric_names =['Dice (All)', 'Dice (Tumor)', 'IoU', 'Recall', 'Precision', 'Specificity']
metric_values = [dice_all, dice_tumor_only, iou_all, recall, precision, specificity]
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan']

plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values, color=colors, width=0.55, edgecolor='black', alpha=0.85)
plt.ylim(0, 110)
plt.ylabel('Score (%)', fontsize=12)
plt.title('U-Net Metrics', fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("results/metrics_summary_unet.png", dpi=300)
print("Saved metrics chart to: results/metrics_summary_unet.png")
plt.close()

# Plot 2: Visualization with overlay
fig, axes = plt.subplots(len(sample_images), 4, figsize=(16, 4 * len(sample_images)))

for idx in range(len(sample_images)):
    img = sample_images[idx]
    true_mask = sample_masks[idx]
    pred_mask = sample_preds[idx]

    # Column 1: Original MRI Image
    axes[idx, 0].imshow(img)
    axes[idx, 0].set_title("Original MRI Scan", fontsize=11)
    axes[idx, 0].axis("off")

    # Column 2: Ground Truth Radiologist Mask
    axes[idx, 1].imshow(true_mask, cmap="gray")
    axes[idx, 1].set_title("Ground Truth Mask", fontsize=11)
    axes[idx, 1].axis("off")

    # Column 3: Predicted Mask from U-Net
    axes[idx, 2].imshow(sample_preds[idx], cmap="gray")
    axes[idx, 2].set_title(f"U-Net Prediction (Threshold: {THRESHOLD})", fontsize=11)
    axes[idx, 2].axis("off")

    # Column 4: Overlay (green = truth, red/yellow = AI)
    overlay = img.copy()
    # Green channel for true tumor
    overlay[true_mask == 1] = [0, 1, 0]
    # Red channel for AI prediction, yellow for overlap
    overlay[pred_mask == 1] = [1, 0, 0]

    axes[idx, 3].imshow(overlay)
    axes[idx, 3].set_title("Overlay (green:true, yellow:agree)", fontsize=11)
    axes[idx, 3].axis("off")

plt.tight_layout()
plt.savefig("results/monai_unet_evaluation_overlay.png", dpi=300)
print("Saved qualitative showcase to: results/monai_unet_evaluation_overlay.png")
plt.close()