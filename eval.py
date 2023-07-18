from config.configGAN import ConfigGAN
from model.latentGAN import GAN
from data.lgan_dataset import get_dataloader as get_dataloader_for_gan
import torch
from model.autoencoder import AutoEncoder
from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader_from_tensor as get_dataloader_for_ae_decoder
from data.ae_dataset import get_dataloader as get_dataloader_for_ae


def compute_cosine_similarity(pred, true):
    return None


if __name__ == '__main__':
    # Load in checkpoints
    gan_ckpt_file = 'lightning_logs/0718/105055/checkpoints/best_gan.ckpt'
    gan_cfg = ConfigGAN('test')
    gan = GAN.load_from_checkpoint(gan_ckpt_file, cfg=gan_cfg)
    gan.eval()
    test_loader_for_gan = get_dataloader_for_gan(gan_cfg)

    ae_ckpt_file = 'lightning_logs/0717/170857/checkpoints/best.ckpt'
    ae_cfg = ConfigAE('test')
    ae = AutoEncoder.load_from_checkpoint(ae_ckpt_file, cfg=ae_cfg)
    ae.eval()

    # Generate z_hat from central part + noise
    batch_z_hat = []
    batch_z = []
    for batch_idx, batch in enumerate(test_loader_for_gan):
        real_z, c_emb = batch
        real_z = real_z.view(real_z.size(0), -1)
        batch_z.append(real_z)

        # Sample noise
        bs = real_z.shape[0]
        noise = torch.randn(bs, gan_cfg.n_dim)
        # Append central part embedding
        noise = torch.cat((c_emb, noise), 1)   # [bs, n_dim+z_dim]

        z_hat = gan.forward(noise.cuda())
        batch_z_hat.append(z_hat)

    all_z_hat = torch.cat(batch_z_hat, dim=0)
    all_z = torch.cat(batch_z, dim=0)
    print(all_z_hat.shape)   # [10377, 512]
    assert all_z_hat.shape == all_z.shape, "all_z_hat and all_z are not having the same shape"

    # Input generated z_hat to the trained Decoder in AE
    test_loader_for_ae_decoder = get_dataloader_for_ae_decoder(all_z_hat, ae_cfg)
    test_loader_for_ae = get_dataloader_for_ae('test', ae_cfg)

    for batch_idx, batch in enumerate(test_loader_for_ae_decoder):
        out = ae.decode(batch[0].cuda())   # batch is a len=1 list (inp, tgt?), out has the shape of [bs, 5120]
        print(out.shape)
        bs = out.shape[0]
        for i in range(bs):
            predicted_seq = out[i]   # [5120,]
            # predicted_parts_emb = torch.split(predicted_seq, 512, 0)   # tuple of 10x [512,]
            predicted_parts_emb = predicted_seq.reshape((-1, 512))


    # compute_cosine_similarity()

