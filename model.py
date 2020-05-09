import torch
import torch.nn as nn
from utils import Flatten, UnFlatten
import numpy as np
import torch.distributions as td

class BetaVAE(nn.Module):
    def __init__(self, filters=[8], latent=5):
        super(BetaVAE, self).__init__()

        self.filters = filters
        self.latent = latent

        # Encoder
        enc_layers = [nn.Conv2d(1, filters[0], kernel_size=3, padding=1, stride=2),
                      nn.BatchNorm2d(filters[0]),
                      nn.ReLU()]
        for i in range(len(filters)-1):
            enc_layers.extend([nn.Conv2d(filters[i], filters[i+1], kernel_size=3, padding=1, stride=2),
                              nn.BatchNorm2d(filters[i+1]),
                              nn.ReLU()])
        enc_layers.extend([nn.Conv2d(filters[-1], 1, kernel_size=3, padding=1, stride=2),
                      nn.BatchNorm2d(1),
                      nn.ReLU(),
                      Flatten()])
        self.encoder = nn.Sequential(*enc_layers)

        # Latent
        self.conv_out_dim = int(28 / 2**(len(filters)+1))**2
        self.mu = nn.Linear(self.conv_out_dim, latent)
        self.lv = nn.Linear(self.conv_out_dim, latent)
        self.conv_prep = nn.Sequential(nn.Linear(latent, self.conv_out_dim), nn.ReLU())

        # Decoder
        dec_layers = [nn.ConvTranspose2d(1, filters[-1], kernel_size=3, padding=1, stride=2),
                      nn.BatchNorm2d(filters[-1]),
                      nn.ReLU()]
        for i in reversed(range(len(filters)-1)):
            dec_layers.extend([nn.ConvTranspose2d(filters[i], filters[i + 1], kernel_size=3, padding=1, stride=2),
                               nn.BatchNorm2d(filters[i + 1]),
                               nn.ReLU()])
        dec_layers.extend([nn.ConvTranspose2d(filters[-1], 1, kernel_size=4, padding=0, stride=2), nn.Sigmoid()])
        self.decoder = nn.Sequential(*dec_layers)

    def BottomUp(self, x):
        out = self.encoder(x)
        mu, lv = self.mu(out), self.lv(out)
        return mu, lv
    def reparameterize(self, mu, lv):
        std = lv.mul(0.5).exp()
        z = torch.distributions.Normal(mu, std).rsample()
        return z

    def TopDown(self, x):
        z = self.conv_prep(x)
        unflatten_dim = int(np.sqrt(self.conv_out_dim))
        z = z.view(x.shape[0], 1, unflatten_dim, unflatten_dim)
        out = self.decoder(z)
        return out

    def forward(self, x):
        mu, lv = self.BottomUp(x)
        z = self.reparameterize(mu, lv)
        out = self.TopDown(z)
        return torch.sigmoid(out)

    def calc_loss(self, x, beta):
        mu, lv = self.BottomUp(x)
        z = self.reparameterize(mu, lv)
        out = self.TopDown(z)

        p_x = td.Normal(loc=1, scale=1)
        q_zGx  = td.Normal(loc=mu, scale=lv.mul(0.5).exp())
        kl = td.kl_divergence(q_zGx, p_x).sum() / x.shape[0]

        X_unnormalized = x*0.3081 + 0.1307
        nll = td.Bernoulli(logits=out).log_prob(X_unnormalized).sum() / x.shape[0]

        return nll + kl*beta, kl, nll






