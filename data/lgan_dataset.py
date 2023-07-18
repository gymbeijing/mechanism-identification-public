from torch.utils.data import Dataset, DataLoader
import torch
import random


def get_dataloader(config, shuffle=None):
    is_shuffle = config.is_train=='train' if shuffle is None else shuffle

    dt = LGANDataset(config)
    # is_shuffle = False
    dataloader = DataLoader(dt, batch_size=config.args.batch_size, shuffle=is_shuffle)
    return dataloader


class LGANDataset(Dataset):
    def __init__(self, cfg):
        super(LGANDataset, self).__init__()
        self.cfg = cfg

        self.z = torch.load(self.cfg.args.z_file)

        self.c_emb = torch.load(self.cfg.args.c_emb_file)   # [42919, 512]
        self.num_c = self.c_emb.shape[0]

    def __getitem__(self, idx):
        c_idx = random.randint(0, self.num_c-1)
        return self.z[idx], self.c_emb[c_idx]

    def __len__(self):
        return self.z.shape[0]


