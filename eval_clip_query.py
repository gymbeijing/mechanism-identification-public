from config.configGAN import ConfigGAN
from model.latentWGANmse import WGAN
from data.lgan_dataset import get_dataloader as get_dataloader_for_gan
from data.lgan_dataset import get_dataloader_from_tensor as get_dataloader_for_query
import torch
from model.autoencoder import AutoEncoder
from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader_from_tensor as get_dataloader_for_ae_decoder
from torch import nn
from torch import linalg as LA


def save_tensor(t, dest):
    torch.save(t, dest)

    return


if __name__ == '__main__':
    # Load in checkpoints
    # gan_ckpt_file = 'lightning_logs/0718/105055/checkpoints/best_gan.ckpt'
    # gan_ckpt_file = 'lightning_logs/0724/204251/checkpoints/best_gan-v9.ckpt'   # WGAN
    gan_ckpt_file = 'lightning_logs/0801/161839/checkpoints/last.ckpt'  # WGAN w/ mse
    gan_cfg = ConfigGAN('test')
    gan = WGAN.load_from_checkpoint(gan_ckpt_file, cfg=gan_cfg)
    gan.eval()
    # Load query embeddings into testloader for (W)GAN
    queries = torch.load('../processed_data/clip_query_embs.pt')
    test_loader_for_gan = get_dataloader_for_query(queries, gan_cfg)

    ae_ckpt_file = '../lightning_logs/0717/170857/checkpoints/best.ckpt'
    ae_cfg = ConfigAE('test')
    ae = AutoEncoder.load_from_checkpoint(ae_ckpt_file, cfg=ae_cfg)
    ae.eval()

    # Generate z_hat from central part + noise
    batch_z_hat = []
    batch_z = []  # not used
    for batch_idx, batch_query in enumerate(test_loader_for_gan):

        # Sample noise
        bs = batch_query.shape[0]
        noise = torch.randn(bs, gan_cfg.n_dim)
        # Append central part embedding
        noise = torch.cat((batch_query, noise), 1)  # [bs, n_dim+z_dim]

        z_hat = gan.forward(noise.cuda())
        batch_z_hat.append(z_hat)

    all_z_hat = torch.cat(batch_z_hat, dim=0)

    # Input generated z_hat to the trained Decoder in AE
    test_loader_for_ae_decoder = get_dataloader_for_ae_decoder(all_z_hat, ae_cfg)

    # Store the predicted sequence for each data point
    rec_seq = []
    rec_norm = []
    for batch_idx, batch in enumerate(test_loader_for_ae_decoder):
        out = ae.decode(batch[0].cuda())  # batch is a len=1 list (inp, tgt?), out has the shape of [bs, 5120]
        # print(out.shape)
        bs = out.shape[0]
        for i in range(bs):
            predicted_seq = out[i]  # [5120,]
            # predicted_parts_emb = torch.split(predicted_seq, 512, 0)   # tuple of 10x [512,]
            predicted_seq = predicted_seq.reshape((-1, 512))  # [10, 512]
            rec_seq.append(predicted_seq.cpu())  # 10377
            rec_norm.append(LA.norm(predicted_seq, dim=1))  # [10,]?

    rec_seq = torch.stack(rec_seq, dim=0)  # Has randomness because of the noise
    save_tensor(rec_seq, "../model_outputs/query_seq_0801_161839_last.pt")
