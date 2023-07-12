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
            nn.Linear(h_dim, 1),
            nn.Sigmoid()   # Another option to map it to [0, 1] is .argmax()
        )

    def forward(self, inputs):
        output = self.decoder(inputs)
        return output


class GAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.G = Generator(cfg.n_dim, cfg.h_dim, cfg.z_dim)
        self.D = Discriminator(cfg.h_dim, cfg.z_dim)
        self.validation_step_outputs = []
        self.lr = cfg.args.lr
        self.n_dim = cfg.n_dim
        self.h_dim = cfg.h_dim
        self.z_dim = cfg.z_dim

        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optimizerG = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        optimizerD = torch.optim.Adam(self.D.parameters(), lr=self.lr)

        return optimizerG, optimizerD

    def training_step(self, train_batch, batch_idx):
        real_data = train_batch
        real_data = real_data.view(real_data.size(0), -1)

        optimizer_g, optimizer_d = self.optimizers()

        # Sample noise
        bs = real_data.shape[0]
        noise = torch.randn(bs, self.n_dim)
        noise = noise.type_as(real_data)

        # Generate samples
        self.toggle_optimizer(optimizer_g)
        fake_data = self.G(noise)

        # Training target of generated samples of G
        valid = torch.ones(bs, 1)
        valid = valid.type_as(real_data)

        # Train G to generate close-to-real samples
        g_loss = F.binary_cross_entropy(self.D(fake_data), valid)
        self.log("g_loss", g_loss, on_epoch=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train D
        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(bs, 1)
        valid = valid.type_as(real_data)

        real_loss = F.binary_cross_entropy(self.D(real_data), valid)

        fake = torch.zeros(bs, 1)
        fake = fake.type_as(real_data)

        fake_loss = F.binary_cross_entropy(self.D(fake_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, on_epoch=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        return

