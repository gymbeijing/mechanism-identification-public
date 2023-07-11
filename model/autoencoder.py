import torch.nn as nn
import pytorch_lightning as pl   # version: 2.0.4
# import lightning as L
import torch.optim
import torch.nn.functional as F


class AutoEncoder(pl.LightningModule):
	def __init__(self, cfg):
	# def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(cfg.dim_emb, cfg.dim_emb//2),   # 5120 -> 2560
			nn.LeakyReLU(),
			nn.Linear(cfg.dim_emb//2, cfg.dim_emb//4),   # 2560 -> 1280
			nn.LeakyReLU(),
			nn.Linear(cfg.dim_emb//4, cfg.dim_z),   # 1280 -> 512
			)
		self.decoder = nn.Sequential(
			nn.Linear(cfg.dim_z, cfg.dim_emb//4),   # 512 -> 1280
			nn.LeakyReLU(),
			nn.Linear(cfg.dim_emb//4, cfg.dim_emb//2),   # 1280 -> 2560
			nn.LeakyReLU(),
			nn.Linear(cfg.dim_emb//2, cfg.dim_emb),   # 2560 -> 5120
		)

		self.validation_step_outputs = []

	def forward(self, x):
		z = self.encoder(x)
		out = self.decoder(z)

		return out

	def encode(self, x):
		z = self.encoder(x)

		return z

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

		return optimizer

	def training_step(self, train_batch, batch_idx):
		x = train_batch
		x = x.view(x.size(0), -1)
		x_hat = self.forward(x)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss, on_epoch=True)

		return loss

	def validation_step(self, val_batch, batch_idx):
		x = val_batch
		x = x.view(x.size(0), -1)
		x_hat = self.forward(x)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)
		self.validation_step_outputs.append(loss)

		return loss

	def on_validation_epoch_end(self):
		val_losses = torch.stack(self.validation_step_outputs)
		mean_val_loss = val_losses.mean()
		self.validation_step_outputs.clear()
		self.log('mean_val_loss', mean_val_loss)
		return {"mean_val_loss": mean_val_loss}


if __name__ == '__main__':
	model = AutoEncoder()
	trainer = pl.Trainer()

