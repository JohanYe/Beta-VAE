import torch
import torch.nn as nn
from utils import Flatten, UnFlatten, cycle_interval
import numpy as np
import torch.distributions as td

class ResidualEncoderBlock(nn.Module):
    # Consider addring gated resnet block instead
    # block_type is a string specifying the structure of the block, where:
    #         a = activation
    #         b = batch norm
    #         c = conv layer
    #         d = dropout.
    # For example, bacd (batchnorm, activation, conv, dropout).
    # TODO: ADDTT uses different number of filters in inner, should we consider that? I've only allowed same currently.

    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), kernel_size=3, block_type=None, dropout=None, stride=2):
        super(ResidualEncoderBlock, self).__init__()

        assert all(c in 'abcd' for c in block_type)
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.kernel_size = kernel_size
        self.block_type = block_type
        self.dropout = dropout
        self.stride = stride

        self.pre_conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2, stride=stride)
        res = []  # Am considering throwing these if statements into separate function
        for character in block_type:
            if character == 'a':
                res.append(nonlin)
            elif character == 'b':
                res.append(nn.BatchNorm2d(c_out))
            elif character == 'c':
                res.append(
                    nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2)
                )
            elif character == 'd':
                res.append(nn.Dropout2d(dropout))
        self.res = nn.Sequential(*res)
        self.post_conv = None  # TODO: Ensure this should not be implemented, consult ADDTT

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res(x) + x
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class ResidualDecoderBlock(nn.Module):
    # Consider addring gated resnet block instead
    # block_type is a string specifying the structure of the block, where:
    #         a = activation
    #         b = batch norm
    #         c = conv layer
    #         d = dropout.
    # For example, bacd (batchnorm, activation, conv, dropout).
    # TODO: ADDTT uses different number of filters in inner, should we consider that? I've only allowed same currently.

    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), kernel_size=3, block_type=None, dropout=None, stride=2):
        super(ResidualDecoderBlock, self).__init__()

        assert all(c in 'abcd' for c in block_type)
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.kernel_size = kernel_size
        self.block_type = block_type
        self.dropout = dropout
        self.stride = stride

        self.pre_conv = nn.ConvTranspose2d(
            c_in, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2, stride=stride, output_padding=1)
        res = []  # Am considering throwing these if statements into separate function
        for character in block_type:
            if character == 'a':
                res.append(nonlin)
            elif character == 'b':
                res.append(nn.BatchNorm2d(c_out))
            elif character == 'c':
                res.append(
                    nn.ConvTranspose2d(c_out, c_out, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
                )
            elif character == 'd':
                res.append(nn.Dropout2d(dropout))
        self.res = nn.Sequential(*res)
        self.post_conv = None  # TODO: Ensure this should not be implemented, consult ADDTT

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res(x) + x
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class BetaVAE_conv(nn.Module):
    def __init__(self, filters=[32, 64, 128], latent=5, block_type='cabd', drop_rate=0.1, MNIST=False):
        super(BetaVAE_conv, self).__init__()

        self.filters = filters
        self.latent = latent
        self.img_dim = 28 if MNIST else 64
        self.block_type = block_type

        # Encoder
        enc_layers = [ResidualEncoderBlock(1, filters[0], kernel_size=3, block_type=block_type, dropout=drop_rate)]
        for i in range(len(filters) - 2):
            enc_layers.append(ResidualEncoderBlock(filters[i],
                                                   filters[i + 1],
                                                   block_type=block_type,
                                                   dropout=drop_rate))
        enc_layers.extend([ResidualEncoderBlock(filters[i+1], filters[i+2], block_type=block_type, dropout=drop_rate),
                           Flatten()])
        self.encoder = nn.Sequential(*enc_layers)

        # Latent
        self.conv_out_dim = int((self.img_dim / 2 ** (len(filters))) ** 2 * filters[-1])
        self.mu = nn.Linear(self.conv_out_dim, latent)
        self.lv = nn.Linear(self.conv_out_dim, latent)
        self.conv_prep = nn.Sequential(nn.Linear(latent, self.conv_out_dim), nn.ReLU())

        # Decoder
        dec_layers = [ResidualDecoderBlock(filters[-1], filters[-1], kernel_size=3, block_type=block_type, dropout=drop_rate)]
        for i in reversed(range(1, len(filters)-1)):
            dec_layers.append(ResidualDecoderBlock(filters[i],
                                                   filters[i - 1],
                                                   block_type=block_type,
                                                   dropout=drop_rate))
        dec_layers.append(ResidualDecoderBlock(filters[0], 1, kernel_size=3, block_type=block_type, dropout=drop_rate))
        self.decoder = nn.Sequential(*dec_layers)

    def BottomUp(self, x):
        out = self.encoder(x)
        mu, lv = self.mu(out), self.lv(out)
        return mu, lv

    def reparameterize(self, mu, lv):
        std = lv.mul(0.5).exp()
        z = td.Normal(mu, std).rsample()
        return z

    def TopDown(self, x):
        z = self.conv_prep(x)
        unflatten_dim = int(np.sqrt(self.conv_out_dim / self.filters[-1]))
        z = z.view(x.shape[0], self.filters[-1], unflatten_dim, unflatten_dim)
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
        out = torch.sigmoid(self.TopDown(z))

        # zeros = torch.zeros_like(mu).detach()
        # ones = torch.ones_like(lv).detach()
        # p_x = td.Normal(loc=zeros, scale=ones)
        # q_zGx = td.Normal(loc=mu, scale=lv.mul(0.5).exp())
        # kl = td.kl_divergence(q_zGx, p_x).sum()# / x.shape[0]

        # x = x*0.3081 + 0.1307
        # nll = td.Bernoulli(logits=out).log_prob(x).sum() / x.shape[0]
        # BCEWithLogitsLoss because binary_cross_entropy_with_logits will not accepts reduction = none
        # nll = -nn.BCEWithLogitsLoss(reduction='none')(out, x).sum()# / x.shape[0]

        nll = -nn.functional.binary_cross_entropy(out, x, reduction='sum') / x.shape[0]
        kl = (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) + 1e-5) / x.shape[0]
        # print(kl, nll, out.min(), out.max())

        return -nll + kl * beta, kl, nll

    def LT_fitted_gauss_2std(self, x,num_var=5, num_traversal=5):
        # Cycle linearly through +-2 std dev of a fitted Gaussian.
        mu, lv = self.BottomUp(x)

        images = []
        for i, batch_mu in enumerate(mu[:num_var]):
            images.append(torch.sigmoid(self.TopDown(batch_mu.unsqueeze(0))))
            for latent_var in range(batch_mu.shape[0]):
                new_mu = batch_mu.unsqueeze(0).repeat([num_traversal, 1])
                loc = mu[:, latent_var].mean()
                total_var = lv[:, latent_var].exp().mean() + mu[:, latent_var].mean()
                scale = total_var.sqrt()
                new_mu[:, latent_var] = cycle_interval(batch_mu[latent_var], num_traversal,
                                                       loc - 2 * scale, loc + 2 * scale)
                print(new_mu.shape)
                images.append(torch.sigmoid(self.TopDown(new_mu)))
        return images




class BetaVAE_Linear(nn.Module):
    def __init__(self, n_hidden=[256, 64], latent=5):
        super(BetaVAE_Linear, self).__init__()

        self.n_hidden = n_hidden
        self.latent = latent

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, n_hidden[0]), nn.ReLU(),
            nn.Linear(n_hidden[0], n_hidden[1]), nn.ReLU(),
        )

        # Latent
        self.mu = nn.Linear(n_hidden[-1], latent)
        self.lv = nn.Linear(n_hidden[-1], latent)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent, n_hidden[1]), nn.ReLU(),
            nn.Linear(n_hidden[1], n_hidden[0]), nn.ReLU(),
            nn.Linear(n_hidden[0], 784)
        )

    def BottomUp(self, x):
        out = self.encoder(x)
        mu, lv = self.mu(out), self.lv(out)
        return mu, lv

    def reparameterize(self, mu, lv):
        std = lv.mul(0.5).exp()
        z = td.Normal(mu, std).rsample()
        return z

    def TopDown(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        mu, lv = self.BottomUp(x)
        z = self.reparameterize(mu, lv)
        out = self.TopDown(z)
        return torch.sigmoid(out)

    def calc_loss(self, x, beta):
        x = x.view(x.shape[0], -1)
        mu, lv = self.BottomUp(x)
        z = self.reparameterize(mu, lv)
        out = self.TopDown(z)

        p_x = td.Normal(loc=0, scale=1)
        q_zGx = td.Normal(loc=mu, scale=lv.mul(0.5).exp())
        kl = td.kl_divergence(q_zGx, p_x).sum() / x.shape[0]

        # x = x*0.3081 + 0.1307
        nll = td.Bernoulli(logits=out).log_prob(x).sum() / x.shape[0]
        # print(kl, nll)

        return -nll + kl * beta, kl, nll

    def LT_fitted_gauss_2std(self, x,num_var=5, num_traversal=5):
        # Cycle linearly through +-2 std dev of a fitted Gaussian.
        x = x.view(x.shape[0], -1)
        mu, lv = self.BottomUp(x)

        images = []
        for i, batch_mu in enumerate(mu[:num_var]):
            images.append(torch.sigmoid(self.TopDown(batch_mu)).unsqueeze(0))
            for latent_var in range(batch_mu.shape[0]):
                new_mu = batch_mu.unsqueeze(0).repeat([num_traversal, 1])
                loc = mu[:, latent_var].mean()
                total_var = lv[:, latent_var].exp().mean() + mu[:, latent_var].mean()
                scale = total_var.sqrt()
                new_mu[:, latent_var] = cycle_interval(batch_mu[latent_var], num_traversal,
                                                       loc - 2 * scale, loc + 2 * scale)
                images.append(torch.sigmoid(self.TopDown(new_mu)))
        return images
