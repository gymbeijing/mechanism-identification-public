from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader
from model.autoencoder import AutoEncoder
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import time


def main():
    cfg = ConfigAE('train')
    model = AutoEncoder(cfg)
    month_day = time.strftime('%m%d')
    hour_min_second = time.strftime('%H%M%S')
    tb_logger = TensorBoardLogger('lightning_logs',
                                  name=month_day,
                                  version=hour_min_second)
    trainer = pl.Trainer(accelerator="gpu", max_epochs=cfg.args.max_epochs, logger=tb_logger)
    train_loader = get_dataloader('train', cfg)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
