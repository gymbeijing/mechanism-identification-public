import torch.nn as nn
import torch

class Encoder(nn.Module):
	def __init__(self, cfg):
		super().__init__()

		self.encode = nn.Sequential(
		                nn.Linear(cfg.dim_emb, cfg.dim_emb//2),   # 5120 -> 2560
						nn.LeakyReLU(),
						nn.Linear(cfg.dim_emb//2, cfg.dim_emb//4),   # 2560 -> 1280
						nn.LeakyReLU(),
						nn.Linear(cfg.dim_emb//4, cfg.dim_z),   # 1280 -> 512
						nn.LeakyReLU()
					   )

	def forward(self, parts_emb):
		z = self.encode(parts_emb)

		return z


class Decoder(nn.Module):
	def __init__(self, cfg):
		super(Decoder, self).__init__()

		self.decode = nn.Sequential(
		                nn.Linear(cfg.dim_z, cfg.dim_emb//4),   # 512 -> 1280
						nn.LeakyReLU(),
						nn.Linear(cfg.dim_emb//4, cfg.dim_emb//2),   # 1280 -> 2560
						nn.LeakyReLU(),
						nn.Linear(cfg.dim_emb//2, cfg.dim_emb),   # 2560 -> 5120
						nn.LeakyReLU()
					   )

	def forward(self, z):
		out = self.decode(z)

		return out
