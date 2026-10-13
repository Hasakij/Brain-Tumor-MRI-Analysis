from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A

from unet_model import Unet
from dataset_unet import LGGDataset

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50

# Augmentation for training only
train_transform = A.Compose([
	A.HorizontalFlip(p=0.5),
	A.RandomRotate90(p=0.5),
	A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
])

class BCEDiceLoss(nn.Module):
	def __init__(self, pos_weight=5.0, smooth=1.0):
		super().__init__()
		self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
		self.smooth = smooth

	def forward(self, logits, targets):
		bce = self.bce(logits, targets)

		probs = torch.sigmoid(logits)
		probs_flat = probs.view(probs.size(0), -1)
		targets_flat = targets.view(targets.size(0), -1)

		intersection = torch.sum(probs_flat * targets_flat, dim=1)
		cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

		dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
		dice_loss = 1.0 - torch.mean(dice)
		return bce + dice_loss
  
# Data preparation
data_dir = Path("/home/adam/.cache/kagglehub/datasets/mri_unet/lgg-mri-segmentation/kaggle_3m")
all_images = [p for p in data_dir.rglob("*.tif") if "_mask" not in p.stem]

train_paths, val_paths = train_test_split(all_images, test_size=0.2, random_state=42)

train_ds = LGGDataset(image_paths=train_paths, transform=train_transform)
val_ds = LGGDataset(image_paths=val_paths, transform=None)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

# Model, optimizer, scheduler
model = Unet(in_channels=3, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = BCEDiceLoss(pos_weight=5.0)
scaler = torch.amp.GradScaler('cuda')

print(f"Starting training on: {device}")

best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Training loop
for epoch in range(EPOCHS):
	model.train()
	train_losses = torch.tensor(0.0, device=device)

	for batch in train_loader:
		images = batch['image'].to(device, non_blocking=True)
		masks = batch['mask'].to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		with torch.amp.autocast('cuda'):
			preds = model(images)
			loss = criterion(preds, masks)
		
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		train_losses += loss.detach()

	# Validation
	model.eval()
	val_losses = torch.tensor(0.0, device=device)

	with torch.no_grad():
		with torch.amp.autocast('cuda'):
			for batch in val_loader:
				images = batch['image'].to(device, non_blocking=True)
				masks = batch['mask'].to(device, non_blocking=True)
				preds = model(images)
				loss = criterion(preds, masks)
				val_losses += loss.detach()
	
	scheduler.step()

	epoch_train_loss = (train_losses / len(train_loader)).item()
	epoch_val_loss = (val_losses / len(val_loader)).item()

	# Model checkpoint and early stopping
	if epoch_val_loss < best_val_loss:
		best_val_loss = epoch_val_loss
		patience_counter = 0
		torch.save(model.state_dict(), "best_unet.pth")
		print(f"Saved best model in epoch {epoch+1:02d}. Val loss: {epoch_val_loss:.4f}")
	else:
		patience_counter += 1
		if patience_counter >= patience:
			print(f"No progress for {patience} epochs. Early stopping.")
			break

	print(F"Epoch [{epoch+1:02d}/{EPOCHS}] | Train loss: {epoch_train_loss:.4f} | Val loss: {epoch_val_loss:.4f}\n")

print("Training finished.")