from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class LGGDataset(Dataset):
	def __init__(self, image_paths=None, root_dir=None, transform=None):
		self.transform = transform
		
		if image_paths is not None:
			self.image_paths = image_paths
		elif root_dir is not None:
			self.image_paths = [p for p in self.root_dir.rglob("*.tif") if "_mask" not in p.stem]
		else:
			raise ValueError("Provide either image_paths or root_dir")

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img_path = self.image_paths[idx]
		mask_path = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")


		# Load the images and masks
		image = cv2.imread(str(img_path))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

		# Augmentation (image and mask)
		if self.transform is not None:
			augmented = self.transform(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']

		# Preprocessing
		img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
		mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

		# Binary classification
		label = (mask_tensor > 0).any().long()

		return {
			'image': img_tensor,
			'mask': mask_tensor,
			'label': label
		}