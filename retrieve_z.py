import torch
from model.autoencoder import AutoEncoder
from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader


def save_tensor(t, dest):
    torch.save(t, dest)


if __name__ == '__main__':
    ckpt_file = 'lightning_logs/0711/083732/checkpoints/best.ckpt'
    cfg = ConfigAE('train')
    model = AutoEncoder.load_from_checkpoint(ckpt_file, cfg=cfg)
    model.eval()
    train_loader = get_dataloader('train', cfg)

    # Save all z in the training phase
    # batch_z = []
    # for batch_idx, batch in enumerate(train_loader):
    #     z = model.encode(batch.to("cuda"))
    #     batch_z.append(z)
    #
    # all_z = torch.cat(batch_z, dim=0)
    # # print(all_z.shape)
    # path = "model_outputs/z_train.pt"
    # save_tensor(all_z, path)

    # Save all reconstructed inputs in the training phase
    batch_out = []
    for batch_idx, batch in enumerate(train_loader):
        out = model.forward(batch.to("cuda"))
        batch_out.append(out)

    all_out = torch.cat(batch_out, dim=0)
    # print(all_z.shape)
    path = "model_outputs/out_train.pt"
    save_tensor(all_out, path)

