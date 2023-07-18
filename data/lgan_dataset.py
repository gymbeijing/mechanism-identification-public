from torch.utils.data import Dataset, DataLoader
import torch


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

        # self.c_emb =

    def __getitem__(self, idx):
        return self.z[idx]

    def __len__(self):
        return self.z.shape[0]


