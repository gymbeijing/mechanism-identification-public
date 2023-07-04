import torch.nn as nn
import torch

class Encoder(nn.Module):
	def __init__(self, cfg):
		super().__init__()

		self.encoder = nn.Linear(cfg.dim_emb, cfg.dim_z)
		self.relu = nn.LeakyReLU(inplace=True)

	def forward(self, parts):
		z = self.encoder(parts)
		z = self.relu(z)

		return z


class Decoder(nn.Module):
	def __init__(self, cfg):
		super(Decoder, self).__init__()

		self.decoder = nn.Linear(cfg.dim_z, cfg.dim_emb)
		self.relu = nn.LeakyReLU(inplace=True)

	def forward(self, z):
		out = self.decoder(z)
		out = self.relu(out)

		return out
