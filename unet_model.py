import torch
import torch.nn as nn

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.block(x)

class Unet(nn.Module):
	def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
		super().__init__()
		# Left side (Encoder)
		self.enc1 = ConvBlock(in_channels, 64)
		self.enc2 = ConvBlock(64, 128)

		self.pool = nn.MaxPool2d(2)

		# Bottom (Bottleneck)
		self.bottleneck = ConvBlock(128, 256)

		# Right side (Decoder)
		self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.dec2 = ConvBlock(256, 128)

		self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.dec1 = ConvBlock(128, 64)

		self.final = nn.Conv2d(64, out_channels, kernel_size=1)

	def forward(self, x):

		# Encoder
		s1 = self.enc1(x)
		p1 = self.pool(s1)

		s2 = self.enc2(p1)
		p2 = self.pool(s2)

		# Bottleneck
		b = self.bottleneck(p2)

		# Decoder + Skip Connections
		u2 = self.up2(b)
		cat2 = torch.cat((s2, u2), dim=1)
		d2 = self.dec2(cat2)

		u1 = self.up1(d2)
		cat1 = torch.cat((s1, u1), dim=1)
		d1 = self.dec1(cat1)

		return self.final(d1)