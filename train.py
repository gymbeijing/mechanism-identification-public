from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader
from model.autoencoder import AutoEncoder
from tqdm import tqdm
import pytorch_lightning as pl


def main():
	cfg = ConfigAE('train')
	model = AutoEncoder(cfg)
	trainer = pl.Trainer(accelerator="gpu", max_epochs=cfg.args.max_epochs)
	train_loader = get_dataloader('train', cfg)
	trainer.fit(model, train_loader)


if __name__ == '__main__':
	main()
