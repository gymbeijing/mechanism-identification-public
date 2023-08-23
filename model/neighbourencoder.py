import torch.nn as nn
import pytorch_lightning as pl   # version: 2.0.4
# import lightning as L
import torch.optim
import torch.nn.functional as F


class NeighbourEncoder(pl.LightningModule):
	def __init__(self, cfg):
	# def __init__(self):
		super().__init__()
		self.decoder = nn.Sequential(
			nn.Linear(cfg.dim_z, cfg.dim_emb//4),   # 512 -> 1280
			nn.Dropout(0.2),
			nn.LeakyReLU(),
			nn.Linear(cfg.dim_emb//4, cfg.dim_emb//2),   # 1280 -> 2560
			nn.Dropout(0.2),
			nn.LeakyReLU(),
			nn.Linear(cfg.dim_emb//2, cfg.dim_emb),   # 2560 -> 5120
		)

		self.validation_step_outputs = []

	def forward(self, x):
		out = self.decoder(x)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, mask = train_batch
		x = x.view(x.size(0), -1)
		c = x[:, :512]
		x_hat = self.forward(c)
		loss = F.mse_loss(x_hat * mask, x * mask)   # default: 'mean'
		self.log('train_loss', loss, on_epoch=True)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, mask = val_batch
		x = x.view(x.size(0), -1)
		c = x[:, :512]
		x_hat = self.forward(c)
		loss = F.mse_loss(x_hat * mask, x * mask)
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
	model = NeighbourEncoder()
	trainer = pl.Trainer()

