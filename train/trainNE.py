from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader
from model.neighbourencoder import NeighbourEncoder
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import time
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


def main():
    cfg = ConfigAE('train')
    model = NeighbourEncoder(cfg)

    # Configure the tensorboard logger
    month_day = time.strftime('%m%d')
    hour_min_second = time.strftime('%H%M%S')
    tb_logger = TensorBoardLogger('lightning_logs',
                                  name=month_day,
                                  version=hour_min_second)

    log_dir = Path(tb_logger.log_dir)
    ckpt_path = log_dir/"checkpoints"
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path,
                                          filename="best",
                                          save_top_k=1,
                                          verbose=True,
                                          monitor="mean_val_loss")
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=cfg.args.max_epochs,
                         logger=tb_logger,
                         callbacks=[checkpoint_callback])
    train_loader = get_dataloader('train', cfg)
    # Should add cfg = ConfigAE('test')
    # And no need to pass 'test' which is already in cfg.is_train
    # But the logic is not affected, since only cfg.batch_size from cfg is used in constructing the test_loader so far
    test_loader = get_dataloader('test', cfg)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
