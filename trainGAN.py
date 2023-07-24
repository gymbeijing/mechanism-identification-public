from config.configGAN import ConfigGAN
from data.lgan_dataset import get_dataloader
from model.latentGAN import GAN
from model.latentWGAN import WGAN
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import time
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


def main():
    cfg = ConfigGAN('train')
    model = WGAN(cfg)

    # Configure the tensorboard logger
    month_day = time.strftime('%m%d')
    hour_min_second = time.strftime('%H%M%S')
    tb_logger = TensorBoardLogger('lightning_logs',
                                  name=month_day,
                                  version=hour_min_second)

    log_dir = Path(tb_logger.log_dir)
    ckpt_path = log_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path,
                                          filename="best_gan",
                                          save_last=True,
                                          save_top_k=10,
                                          verbose=True,
                                          monitor="critic_loss")   # Min. g_loss for GAN, D_cost ("D_loss") for WGAN
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=cfg.args.max_epochs,
                         logger=tb_logger,
                         callbacks=[checkpoint_callback])
    train_loader = get_dataloader(cfg)
    test_cfg = ConfigGAN('test')
    test_loader = get_dataloader(test_cfg)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
