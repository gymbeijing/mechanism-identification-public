import torch.nn as nn
import pytorch_lightning as pl   # version: 2.0.4
import torch.optim
import torch.nn.functional as F
import torch.autograd as autograd


class Generator(nn.Module):
    def __init__(self, n_dim, h_dim, z_dim):
        super().__init__()
        self.generator = nn.Sequential(
            # nn.Linear(n_dim, h_dim),   # w/o central part's emb
            nn.Linear(n_dim + z_dim, h_dim),   # w/ central part's emb
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, z_dim),
            # nn.Tanh(),   # commenting out results in slightly worse performance in terms of the D scores of fake_data
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
        return output


class WGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.G = Generator(cfg.n_dim, cfg.h_dim, cfg.z_dim)
        self.D = Discriminator(cfg.h_dim, cfg.z_dim)
        self.validation_step_outputs = []
        self.lr = cfg.args.lr
        self.n_dim = cfg.n_dim
        self.h_dim = cfg.h_dim
        self.z_dim = cfg.z_dim
        self.gp_lambda = cfg.gp_lambda

        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.scores_real_data = []
        self.scores_fake_data = []
        self.real_z = []
        self.fake_z = []
        self.val_fake_z = []
        self.val_real_z = []
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, noise):
        return self.G(noise)

    def configure_optimizers(self):
        optimizerG = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        optimizerD = torch.optim.Adam(self.D.parameters(), lr=self.lr)

        return optimizerG, optimizerD

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        bs = real_data.shape[0]
        alpha = torch.rand(bs, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda  # LAMBDA

        return gradient_penalty

    def training_step(self, train_batch, batch_idx):
        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.cuda()
        mone = mone.cuda()

        real_data, rand_c_emb = train_batch
        real_data = real_data.view(real_data.size(0), -1)
        real_data.requires_grad_(True)   # necessary?
        bs = real_data.shape[0]
        self.real_z.append(real_data)

        optimizer_g, optimizer_d = self.optimizers()

        # Train D
        self.toggle_optimizer(optimizer_d)   # parameters in all the other optimizers are set to requires_grad=False
        self.D.zero_grad()

        # Train with real
        D_real = self.D(real_data)
        D_real = D_real.mean(dim=0, keepdim=True)
        self.scores_real_data.append(D_real)
        self.manual_backward(D_real * mone)   # we want to max. D_real, which is equivalent to min. -D_real

        # Generate samples
        # Sample noise
        noise = torch.randn(bs, self.n_dim)
        noise = noise.type_as(real_data)
        noise = torch.cat((rand_c_emb, noise), 1)  # [bs, n_dim+z_dim], append the central part's emb to the front
        # Generate fake data
        fake_data = self.G(noise)

        # Train with fake
        self.fake_z.append(fake_data)
        D_fake = self.D(fake_data)
        D_fake = D_fake.mean(dim=0, keepdim=True)
        self.scores_fake_data.append(D_fake)
        self.manual_backward(D_fake * one)   # we want to max. -D_fake, which is equivalent to min. D_fake

        gradient_penalty = self.calc_gradient_penalty(self.D, real_data, fake_data)
        self.manual_backward(gradient_penalty)

        # Update parameters in D
        critic_loss = D_real - D_fake   # D wants to max. critic_loss, so that D_fake and D_real can be largely separated, G wants critic loss=0
        D_cost = D_fake - D_real + gradient_penalty   # we want to min. D_cost

        self.log("D_loss", D_cost, on_epoch=True, prog_bar=True)   # Min. D_cost
        self.log("critic_loss", critic_loss, on_epoch=True, prog_bar=True)   # Max. critic loss
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Train G
        self.toggle_optimizer(optimizer_g)   # parameters in all the other optimizers are set to requires_grad=False
        self.G.zero_grad()
        # Re-generate the noise
        noise = torch.randn(bs, self.n_dim)
        noise = noise.type_as(real_data)
        noise = torch.cat((rand_c_emb, noise), 1)  # [bs, n_dim+z_dim], append the central part's emb to the front
        noise.requires_grad_(True)  # necessary?
        # Re-generate fake data
        fake_data = self.G(noise)

        # Train with fake (generated samples)
        G = self.D(fake_data)
        G = G.mean(dim=0, keepdim=True)
        self.manual_backward(G * mone)   # we want to max. G

        # Update parameters in G
        g_loss = G
        G_cost = -G
        self.log("G_loss", G_cost, on_epoch=True, prog_bar=True)   # Min. G_cost
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        return

    def on_train_epoch_end(self):
        self.logger.experiment.add_histogram("Scores for the real data", torch.cat(self.scores_real_data),
                                             self.current_epoch)
        self.scores_real_data.clear()
        self.logger.experiment.add_histogram("Elements of the real latent variable(z)", torch.cat(self.real_z),
                                             self.current_epoch)
        self.real_z.clear()

        self.logger.experiment.add_histogram("Scores for the fake data", torch.cat(self.scores_fake_data),
                                             self.current_epoch)
        self.scores_fake_data.clear()
        self.logger.experiment.add_histogram("Elements of the generated latent variable(z_hat)", torch.cat(self.fake_z),
                                             self.current_epoch)
        self.fake_z.clear()

        return

    def validation_step(self, val_batch, batch_idx):
        real_data, rand_c_emb = val_batch
        real_data = real_data.view(real_data.size(0), -1)
        bs = real_data.shape[0]

        noise = torch.randn(bs, self.n_dim)
        noise = noise.type_as(real_data)
        noise = torch.cat((rand_c_emb, noise), 1)  # [bs, n_dim+z_dim], append the central part's emb to the front
        # Generate fake data
        fake_data = self.G(noise)
        self.val_fake_z.append(fake_data)
        self.val_real_z.append(real_data)

        return

    def on_validation_epoch_end(self):
        val_fake_data = torch.cat(self.val_fake_z).reshape(-1, 512)   # [10377, 512]
        val_real_data = torch.cat(self.val_real_z).reshape(-1, 512)  # [10377, 512]
        first_fake = val_fake_data[0]
        sim_scores_fake = self.cos(first_fake, val_fake_data)
        first_real = val_real_data[0]
        sim_scores_real = self.cos(first_real, val_real_data)
        self.logger.experiment.add_histogram("Histogram of cosine similarity scores within fake z on validation set",
                                             sim_scores_fake.cpu(),
                                             self.current_epoch)
        self.val_fake_z.clear()
        self.logger.experiment.add_histogram("Histogram of cosine similarity scores within real z on validation set",
                                             sim_scores_real.cpu(),
                                             self.current_epoch)
        self.val_real_z.clear()
        return


