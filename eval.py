from config.configGAN import ConfigGAN
from model.latentGAN import GAN
from model.latentWGAN import WGAN
from data.lgan_dataset import get_dataloader as get_dataloader_for_gan
import torch
from model.autoencoder import AutoEncoder
from config.configAE import ConfigAE
from data.ae_dataset import get_dataloader_from_tensor as get_dataloader_for_ae_decoder
from data.ae_dataset import get_dataloader as get_dataloader_for_ae
from torch import nn
from torch import linalg as LA


def compute_cosine_similarity(pred, true):
    assert len(pred) == len(true), "pred and true sequences are not having the same lengths"
    total = len(pred)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = []
    for idx in range(total):
        score = cos(pred[idx], true[idx])
        scores.append(score)

    return scores


def save_tensor(t, dest):
    torch.save(t, dest)


if __name__ == '__main__':
    # Load in checkpoints
    # gan_ckpt_file = 'lightning_logs/0718/105055/checkpoints/best_gan.ckpt'
    gan_ckpt_file = 'lightning_logs/0724/204251/checkpoints/best_gan-v9.ckpt'   # WGAN
    gan_cfg = ConfigGAN('test')
    gan = WGAN.load_from_checkpoint(gan_ckpt_file, cfg=gan_cfg)
    gan.eval()
    # Load in test loader for (W)GAN
    test_loader_for_gan = get_dataloader_for_gan(gan_cfg)

    ae_ckpt_file = 'lightning_logs/0717/170857/checkpoints/best.ckpt'
    ae_cfg = ConfigAE('test')
    ae = AutoEncoder.load_from_checkpoint(ae_ckpt_file, cfg=ae_cfg)
    ae.eval()

    # Generate z_hat from central part + noise
    batch_z_hat = []
    batch_z = []   # not used
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
    test_loader_for_ae = get_dataloader_for_ae('test', ae_cfg)   # input

    # Store the predicted sequence for each data point
    rec_seq = []
    rec_norm = []
    for batch_idx, batch in enumerate(test_loader_for_ae_decoder):
        out = ae.decode(batch[0].cuda())   # batch is a len=1 list (inp, tgt?), out has the shape of [bs, 5120]
        # print(out.shape)
        bs = out.shape[0]
        for i in range(bs):
            predicted_seq = out[i]   # [5120,]
            # predicted_parts_emb = torch.split(predicted_seq, 512, 0)   # tuple of 10x [512,]
            predicted_seq = predicted_seq.reshape((-1, 512))   # [10, 512]
            rec_seq.append(predicted_seq.cpu())   # 10377
            rec_norm.append(LA.norm(predicted_seq, dim=1))   # [10,]?

    # Store the original sequence for each data point
    ori_seq = []
    for batch_idx, batch in enumerate(test_loader_for_ae):
        inp, _ = batch
        bs = inp.shape[0]
        for i in range(bs):
            input_seq = inp[i]
            input_seq = input_seq.reshape((-1, 512))   # [10, 512]
            ori_seq.append(input_seq)

    # find the pad for ori, compute the 'mask'
    eps = 0.2
    threshold = 0.5

    # normalization in need for rec_seq? think not
    sim_scores = compute_cosine_similarity(rec_seq, ori_seq)
    # print(sim_scores[0])   # Mode collapse encountered. Mode collapse fixed by introducing WGAN.

    rec_seq = torch.stack(rec_seq, dim=0)
    ori_seq = torch.stack(ori_seq, dim=0)

    save_tensor(rec_seq, "./model_outputs/rec_seq_0724_204251_v9.pt")

