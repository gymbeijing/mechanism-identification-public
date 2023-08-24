from config.configGAN import ConfigGAN
# from model.latentGAN import GAN
# from model.latentWGAN import WGAN
# from model.latentWGANmse import WGAN
from model.latentWGANnoise import WGAN
from data.lgan_dataset import get_dataloader as get_dataloader_for_gan
import torch
# from model.autoencoder import AutoEncoder
from model.neighbourencoder import NeighbourEncoder
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

    return


if __name__ == '__main__':
    # Load in checkpoints

    ae_ckpt_file = 'lightning_logs/0824/135617/checkpoints/last.ckpt'
    ae_cfg = ConfigAE('test')
    ae = NeighbourEncoder.load_from_checkpoint(ae_ckpt_file, cfg=ae_cfg)
    ae.eval()

    # Input generated z_hat to the trained Decoder in AE
    # test_loader_for_ae_decoder = get_dataloader_for_ae_decoder(all_z_hat, ae_cfg)
    # test_loader_for_ae_decoder = get_dataloader_for_ae_decoder(all_z, ae_cfg)
    test_loader_for_ae = get_dataloader_for_ae('train', ae_cfg)   # input

    # Store the predicted sequence for each data point
    rec_seq = []
    rec_norm = []
    for batch_idx, batch in enumerate(test_loader_for_ae):
        batch_seq, batch_mask = batch
        batch_c = batch_seq[:, :512]
        out = ae.forward(batch_c.cuda())   # batch is a len=1 list (inp, tgt?), out has the shape of [bs, 5120]
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

    # normalization in need for rec_seq? think not
    # sim_scores = compute_cosine_similarity(rec_seq, ori_seq)
    # print(sim_scores[0])   # Mode collapse encountered. Mode collapse fixed by introducing WGAN.

    rec_seq = torch.stack(rec_seq, dim=0)   # Has randomness because of the noise
    ori_seq = torch.stack(ori_seq, dim=0)

    save_tensor(rec_seq, "./model_outputs/rec_seq_0824_135617_last_train.pt")
    save_tensor(ori_seq, "./model_outputs/ori_seq_0824_135617_last_train.pt")

