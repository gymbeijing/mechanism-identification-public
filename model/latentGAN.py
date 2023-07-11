import torch.nn as nn
import pytorch_lightning as pl   # version: 2.0.4
import torch.optim
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_dim, h_dim, z_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(n_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Tanh(),
        )

    def forward(self, noise):
        output = self.generator(noise)
        return output


class Discriminator(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, inputs):
        output = self.decoder(inputs)
        return output.view(-1)


# class latentGAN