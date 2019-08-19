import torch
import torch.nn as nn
import modules
from collections import OrderedDict
import os
import numpy as np
import utils as utils
import torch.nn.functional as F


def deconvout_(in_size, ker, pad, stride, dil):
    # Note, we're making an assumption of squareness
    out_size = stride * (in_size - 1) - 2 * pad + dil * (ker - 1) + 1
    return out_size


def convout_(in_size, ker, pad, stride, dil):
    # Note, we're making an assumption of squareness
    out_size = round((in_size - 1 - dil*(ker - 1) + 2*pad) / stride + 1)
    return out_size


def convout(in_size, seq):
    for m in seq:
        try:
            ker = m.conv.kernel_size[0]
            pad = m.conv.padding[0]
            stride = m.conv.stride[0]
            dil = m.conv.dilation[0]
            in_size = convout_(in_size, ker, pad, stride, dil)
        except:
            pass
    return in_size


def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std) * 0.1
    return mu + std * eps


class TransposeCoordConv(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(TransposeCoordConv, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_dim+2, out_dim, **kwargs)

    def make_grid(self, b, w, h):
        i = torch.tensor(np.linspace(-1, 1, w)).unsqueeze(0)
        j = torch.tensor(np.linspace(-1, 1, h)).unsqueeze(1)
        i = i.repeat(h, 1).unsqueeze(0).unsqueeze(0)
        j = j.repeat(1, w).unsqueeze(0).unsqueeze(0)
        i = i.repeat(b, 1, 1, 1)
        j = j.repeat(b, 1, 1, 1)
        coord = torch.cat((i, j), dim=1).float().cuda()
        return coord

    def forward(self, input):
        coord = self.make_grid(input.shape[0], input.shape[2], input.shape[3])
        x = torch.cat((input, coord), dim=1)
        return self.tconv(x)


class CoordConv(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_dim+2, out_dim, **kwargs)

    def make_grid(self, b, w, h):
        i = torch.tensor(np.linspace(-1, 1, w)).unsqueeze(0)
        j = torch.tensor(np.linspace(-1, 1, h)).unsqueeze(1)
        i = i.repeat(h, 1).unsqueeze(0).unsqueeze(0)
        j = j.repeat(1, w).unsqueeze(0).unsqueeze(0)
        i = i.repeat(b, 1, 1, 1)
        j = j.repeat(b, 1, 1, 1)
        coord = torch.cat((i, j), dim=1).float().cuda()
        return coord

    def forward(self, input):
        coord = self.make_grid(input.shape[0], input.shape[2], input.shape[3])
        x = torch.cat((input, coord), dim=1)
        return self.conv(x)

class SpatialBroadcaster(nn.Module):
    def __init__(self):
        super(SpatialBroadcaster, self).__init__()
        pass

    def make_grid(self, b, w, h):
        i = torch.tensor(np.linspace(-1, 1, w)).unsqueeze(0)
        j = torch.tensor(np.linspace(-1, 1, h)).unsqueeze(1)
        i = i.repeat(h, 1).unsqueeze(0).unsqueeze(0)
        j = j.repeat(1, w).unsqueeze(0).unsqueeze(0)
        i = i.repeat(b, 1, 1, 1)
        j = j.repeat(b, 1, 1, 1)
        coord = torch.cat((i, j), dim=1).float().cuda()
        return coord

    def forward(self, input):
        pass

    def broadcast(self, x, w, h):
        # x is assumed to have dimensions [batch, dim]
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.repeat(1, 1, w, h)
        # grid = self.make_grid(x.shape[0], w, h)
        # x = torch.cat((x, grid), dim=1)
        return x


class BroadcastDecoder(nn.Module):
    def __init__(self, in_size, fc_in_size):
        super(BroadcastDecoder, self).__init__()
        self.broadcast = SpatialBroadcaster()
        self.dcnn = nn.Sequential(
            CoordConv(32, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            CoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            CoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            CoordConv(8, 4, kernel_size=3, padding=0, stride=1),
            nn.PReLU()
        )
        self.in_size = in_size
        self.fc_in_size = fc_in_size

    def forward(self, input):
        pass

    def decode(self, z):
        z = self.broadcast.broadcast(z, self.fc_in_size+14, self.fc_in_size+14)
        z = self.dcnn(z)
        x_hat = z[:, [i for i in range(0, z.shape[1]-1)], :, :]
        m_hat = z[:, [-1], :, :]

        return x_hat, m_hat


class Decoder(nn.Module):
    def __init__(self, in_size, fc_in_size):
        super(Decoder, self).__init__()
        # self.broadcast = SpatialBroadcaster()
        self.mlp = nn.Sequential(
            nn.Linear(32, 256),
            nn.PReLU(),
            nn.Linear(256, 8*fc_in_size**2),
            nn.PReLU()
        )
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, padding=0, stride=1),
            nn.PReLU()
        )
        self.in_size = in_size
        self.fc_in_size = fc_in_size

    def forward(self, input):
        pass

    def decode(self, z):
        # z = self.broadcast.broadcast(z, w+8, h+8)
        z = self.mlp(z)
        z = z.view(z.shape[0], 8, self.fc_in_size, self.fc_in_size)
        z = self.dcnn(z)
        x_hat = z[:, [i for i in range(0, z.shape[1]-1)], :, :]
        m_hat = z[:, [-1], :, :]

        return x_hat, m_hat


class Encoder(nn.Module):
    def __init__(self, in_size):
        super(Encoder, self).__init__()
        in_dim = 3
        self.cnn = nn.Sequential(
            CoordConv(in_dim, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            CoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            CoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU()
        )
        fc_in_size = convout(in_size, self.cnn)
        print('fc_in:', str(8 * fc_in_size**2))
        self.mlp = nn.Sequential(
            nn.Linear(8 * fc_in_size**2, 256),
            nn.PReLU(),
            nn.Linear(256, 32),
            nn.PReLU()
        )
        self.fc_in_size = fc_in_size

    def forward(self, input):
        pass

    def encode(self, x, m):
        z = torch.cat((x, m), dim=1)

        z = self.cnn(x)
        z = z.view(z.shape[0], -1)
        z = self.mlp(z)

        return z


class VAE(nn.Module):
    def __init__(self, in_size):
        super(VAE, self).__init__()
        self.mu_transform = nn.Linear(32, 32)
        self.lv_transform = nn.Linear(32, 32)

        self.encoder = Encoder(in_size)
        fc_in_size = self.encoder.fc_in_size
        self.decoder = BroadcastDecoder(in_size, fc_in_size)

    def forward(self):
        pass

    def encode(self, x, m):
        z = self.encoder.encode(x, m)
        mu = self.mu_transform(z)
        lv = self.lv_transform(z)
        return mu, lv

    def decode(self, mu, lv):
        z = reparam(mu, lv)
        x_hat, m_hat = self.decoder.decode(z)
        return x_hat, m_hat


class AutoEncoder(nn.Module):
    def __init__(self, in_size):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_size)
        fc_in_size = self.encoder.fc_in_size
        self.decoder = BroadcastDecoder(in_size, fc_in_size)

    def forward(self):
        pass

    def encode(self, x, m):
        z = self.encoder.encode(x, m)
        return z

    def decode(self, z):
        x_hat, m_hat = self.decoder.decode(z)
        return x_hat, m_hat


# this is supposed to be some variety of U-Net
class Attender(nn.Module):
    def __init__(self, in_size):
        super(Attender, self).__init__()
        in_dim = 3
        self.cnn = nn.Sequential(
            CoordConv(in_dim + 1, 8, kernel_size=3, padding=0, stride=1),
            # nn.BatchNorm2d(8),
            nn.PReLU(),
            CoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            # .BatchNorm2d(8),
            nn.PReLU(),
            CoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            # nn.BatchNorm2d(8),
            nn.PReLU()
        )
        fc_in_size = convout(in_size, self.cnn)
        self.fc_in_size = fc_in_size
        print('fc_in:', str(8 * fc_in_size ** 2))
        self.mlp1 = nn.Sequential(
            nn.Linear(8 * fc_in_size ** 2, 256),
            nn.InstanceNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.InstanceNorm1d(32),
            nn.PReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 8 * fc_in_size ** 2),
            nn.InstanceNorm1d(8 * fc_in_size ** 2),
            nn.PReLU()
        )
        self.dcnn = nn.Sequential(
            TransposeCoordConv(16, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            TransposeCoordConv(8, 8, kernel_size=3, padding=0, stride=1),
            nn.PReLU(),
            TransposeCoordConv(8, 1, kernel_size=3, padding=0, stride=1)
        )

    def forward(self, x, s):
        a = torch.cat((x, s), dim=1)

        b = self.cnn(a)
        a = b.view(b.shape[0], -1)
        a = self.mlp1(a)
        # a = a + torch.randn_like(a) * 0.05
        a = self.mlp2(a)
        a = a.view(a.shape[0], 8, self.fc_in_size, self.fc_in_size)
        c = torch.cat((a, b), dim=1)
        a = self.dcnn(c)

        # a = a + torch.randn_like(a) * 0.1

        return a


def kldiv(mu, lv):
    kl_div = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return kl_div


class MONet(nn.Module):
    def __init__(self, in_size):
        super(MONet, self).__init__()
        self.mse = nn.L1Loss()

        self.config = dict()

        self.attender = Attender(in_size)
        self.auto = VAE(in_size)

        self.start_epoch = 0

        self.sigmoid = nn.Sigmoid()

        self.num_steps = 5

    def init_s(self):
        pass

    def recur(self, x, s, l):
        # compute the mask
        if l == 0:
            a = self.attender(x, s)
            # a = a + torch.randn_like(a) * 0.1
            a = self.sigmoid(a)

            m = s * a
            # maxes = torch.max(m.view(m.shape[0], m.shape[1], -1), dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            # maxes[maxes==0] = 1.0
            # m /= maxes

            # compute latent
            mu, lv = self.auto.encode(x, m)
            # z = z + torch.randn_like(z) * 0.05

            # reconstruct
            x_hat, m_hat = self.auto.decode(mu, lv)

            # compute next scope image
            s = s * (1 - a)
        else:
            m = s
            mu, lv = self.auto.encode(x, m)
            x_hat, m_hat = self.auto.decode(mu, lv)

        # x_hat = x_hat + torch.randn_like(x_hat) * 0.05

        return x_hat, m_hat, m, s, mu, lv

    def forward(self, **kwargs):
        x = torch.tensor(kwargs['x'])

        s = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).float().cuda()
        x_hat = torch.zeros_like(x)
        x_recon_loss = torch.tensor(0.0).float().cuda()
        m_recon_loss = torch.tensor(0.0).float().cuda()
        m_shape_loss = torch.tensor(0.0).float().cuda()

        num_steps = self.num_steps
        for k in range(num_steps):
            if k < num_steps-1:
                x_k, m_hat, m, s, mu, lv = self.recur(x, s, 0)
            else:
                x_k, m_hat, m, s, mu, lv = self.recur(x, s, 1)
            x_recon_loss += self.mse(m * x, m * x_k)
            m_recon_loss += self.mse(m, m_hat)
            kl_div = kldiv(mu, lv)
            x_hat = x_hat + m * x_k.detach()
            x = x - m.detach() * x_k.detach()

        return {
            'x_hat': x_hat,
            'x_recon_loss': x_recon_loss,
            'm_recon_loss': m_recon_loss,
            'kl_div': kl_div,
        }

    def predict(self, **kwargs):
        x = torch.tensor(kwargs['x'])

        s = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).float().cuda()
        x_hat = torch.zeros_like(x)

        xs = list()
        ms = list()

        num_steps = self.num_steps
        for k in range(num_steps):
            if k < num_steps - 1:
                x_k, m_hat, m, s, mu, lv = self.recur(x, s, 0)
            else:
                x_k, m_hat, m, s, mu, lv = self.recur(x, s, 1)
            x_hat = x_hat + m * x_k.detach()
            x = x - m.detach() * x_k.detach()
            x = x - m.detach() * x_k.detach()
            xs.append(torch.tensor(x_k))
            ms.append(m)

        return {
            'x_hat': x_hat,
            'xs': xs,
            'ms': ms
        }

    def save_model(self, path, filename):
        model = {
            'model': MONet,
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(model, path + filename)

    @staticmethod
    def load_model(path, filename):
        checkpoint = torch.load(path + filename)
        model = checkpoint['model'](**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def set_noise_std(self, new_std):
        pass


class ConvGRU(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(ConvGRU, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()

        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.injection = nn.Conv2d(in_dim, hid_dim, kernel_size=3, padding=0, bias=False)
        self.W_h = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1, bias=False)
        self.W_x = nn.Conv2d(2 * hid_dim, hid_dim, kernel_size=3, padding=1, bias=False)
        self.V_h = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1, bias=False)
        self.V_x = nn.Conv2d(2 * hid_dim, hid_dim, kernel_size=3, padding=1, bias=False)
        self.U_h = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1, bias=False)
        self.U_x = nn.Conv2d(2 * hid_dim, hid_dim, kernel_size=3, padding=1, bias=False)
        # check how to initialize these
        self.b_r = nn.Parameter(torch.zeros(hid_dim, 1, 1) + 0.5)
        self.b_z = nn.Parameter(torch.zeros(hid_dim, 1, 1) + 1)
        self.b_h = nn.Parameter(torch.zeros(hid_dim, 1, 1))

    def forward(self, h, d, e):
        ee = self.injection(e)

        assert ee.shape[1] == self.hid_dim

        if d is None:
            d = torch.zeros(ee.shape[0], self.hid_dim, ee.shape[2], ee.shape[3]).cuda()

        assert d.shape[1] == self.hid_dim

        x = torch.cat((d, ee), dim=1)

        if h is None:
            z = self.sigmoid(self.V_x(x) + self.b_z)
            h_new = self.tanh(self.U_x(x) + self.b_h)
            h = (1 - z) * h_new
        else:
            r = self.sigmoid(self.W_h(h) + self.W_x(x) + self.b_r)
            z = self.sigmoid(self.V_h(h) + self.V_x(x) + self.b_z)
            h_new = self.tanh(r * self.U_h(h) + self.U_x(x) + self.b_h)
            h = (z * h) + ((1 - z) * h_new)
        # print(h.shape)
        assert h.shape[2] == x.shape[2]
        assert h.shape[1] == self.hid_dim
        return h


class GRUED(nn.Module):
    def __init__(self, layer_dims, index, position, noise_std, arglist):
        super(GRUED, self).__init__()
        # this module will hold the variables it needs to in a dictionary
        # it will also have a set of functions
        self.index = index
        self.layer_dims = layer_dims
        self.position = position
        self.noise_std = noise_std

        self.use_bn = True

        # encoding modules
        if self.position is 'first':
            en_indim = self.layer_dims[self.index]
            en_outdim = self.layer_dims[self.index]
        else:
            en_indim = layer_dims[self.index-1]
            en_outdim = layer_dims[self.index]
            # self.en_conv = nn.Conv2d(en_indim, en_outdim, bias=False, **arglist[self.index-1])
            self.en_gru = ConvGRU(en_indim, en_outdim)
        self.en_bn_clean = nn.BatchNorm2d(en_outdim, affine=False)
        self.en_bn_noisy = nn.BatchNorm2d(en_outdim, affine=False)
        self.en_gamma = nn.Parameter(torch.rand(en_outdim, 1, 1))
        self.en_beta = nn.Parameter(torch.rand(en_outdim, 1, 1))
        self.en_nonlin = nn.Tanh()

        # decoding modules
        if self.position is 'last':
            de_indim = self.layer_dims[self.index]
            de_outdim = self.layer_dims[self.index]
        else:
            de_indim = self.layer_dims[self.index+1]
            de_outdim = self.layer_dims[self.index]
            self.de_conv = nn.ConvTranspose2d(de_indim, de_outdim, bias=False, **arglist[self.index])
        self.de_bn = nn.BatchNorm2d(de_outdim, affine=False)
        self.de_gamma = nn.Parameter(torch.rand(de_outdim, 1, 1))
        self.de_beta = nn.Parameter(torch.rand(de_outdim, 1, 1))
        self.ver_dropout = modules.Dropout(0.5)
        self.lat_dropout = modules.Dropout(0.5)
        self.parsig1 = modules.ParamSigmoid()
        self.parsig2 = modules.ParamSigmoid()

    def set_ver_dropout(self, p):
        self.ver_dropout.p = 1-p

    def set_lat_dropout(self, p):
        self.lat_dropout.p = 1-p

    def deconvout(self, in_size):
        if self.position is 'last':
            return in_size
        else:
            # Note, we're making an assumption of squareness
            ker = self.de_conv.kernel_size[0]
            stride = self.de_conv.stride[0]
            pad = self.de_conv.padding[0]
            dil = self.de_conv.dilation[0]
            out_size = stride * (in_size - 1) - 2 * pad + dil * (ker - 1) + 1
            return out_size

    def forward(self, input):
        raise Exception('You should use either the encode or decode functions')

    # This function performs the clean encoding pass of one layer of the ladder network
    def encode_clean(self, variables):
        # print('Clean encoder:', self.index)
        varx = variables[self.index]

        # if first layer (index=0), z_pre_(i) = x
        if self.position is 'first':
            z_pre = variables[self.index]['x']
        else:
            if 'z' not in varx:
                varx['z'] = None
            if 'z_hat_bn' not in varx:
                varx['z_hat_bn'] = None

            z_pre = self.en_gru(varx['z'], varx['z_hat_bn'], variables[self.index-1]['h'])

        # collect batch statistics
        varx['mean'], varx['std'] = modules.get_batch_params(z_pre)

        if self.use_bn:
            varx['z'] = self.en_bn_clean(z_pre)
        else:
            varx['z'] = z_pre

        # if first layer (index=0), h_(i) = z_(i)
        if self.position is 'first':
            varx['h'] = varx['z']
        else:
            # varx['h'] = self.en_nonlin(self.en_gamma * (varx['z'] + self.en_beta))  # original formulation
            varx['h'] = 1.6 * self.en_nonlin(self.en_gamma * varx['z'] + self.en_beta)  # I think this makes more sense

    # This function performs the noisy encoding pass of one layer of the ladder network
    def encode_noisy(self, variables):
        # print('Noisy encoder:', self.index)
        varx = variables[self.index]

        # if first layer (index=0), z_pre_tilda_(i) = x
        if self.position is 'first':
            z_pre_tilda = variables[self.index]['x']
        else:
            if 'z_tilda' not in varx:
                varx['z_tilda'] = None
            if 'z_hat_bn' not in varx:
                varx['z_hat_bn'] = None

            z_pre_tilda = self.en_gru(varx['z_tilda'], varx['z_hat_bn'], variables[self.index - 1]['h_tilda'])

        # we don't record the mean and std here

        noise = self.noise_std * torch.randn_like(z_pre_tilda)
        # if self.index == 0 and self.noise_std > 0:
            # noise += torch.zeros_like(z_pre_tilda).exponential_(1 / self.noise_std)
            # noise += z_pre_tilda * (2 * torch.rand_like(z_pre_tilda) - 1)

        if self.use_bn:
            varx['z_tilda'] = self.en_bn_noisy(z_pre_tilda) + noise
        else:
            varx['z_tilda'] = z_pre_tilda + noise

        # if first layer (index=0), h_tilda_(i) = z_tilda_(i)
        if self.position is 'first':
            varx['h_tilda'] = varx['z_tilda']
        else:
            # varx['h_tilda'] = self.en_nonlin(self.en_gamma * (varx['z_tilda'] + self.en_beta))  # original formulation
            varx['h_tilda'] = 1.6 * self.en_nonlin(self.en_gamma * varx['z_tilda'] + self.en_beta)  # ditto

    def decode(self, variables):
        # print('Decoder:', self.index)
        varx = variables[self.index]
        # print(self.index)

        # if layer layer (index=L), u_(i) = de_batchnorm( h_tilda_(i) )
        if self.position is 'last':
            if self.use_bn:
                u = self.de_bn(variables[self.index]['h_tilda'])
            else:
                u = variables[self.index]['h_tilda']
        else:
            # calculate output padding
            in_shape = variables[self.index + 1]['z_hat'].shape
            # print(in_shape, varx['z_tilda'].shape)
            w_pad = varx['z_tilda'].shape[2] - self.deconvout(in_shape[2])
            h_pad = varx['z_tilda'].shape[3] - self.deconvout(in_shape[3])
            self.de_conv.output_padding = (w_pad, h_pad)
            # print(w_pad, h_pad)
            if self.use_bn:
                u = self.ver_dropout(self.de_bn(self.de_conv(variables[self.index + 1]['z_hat'])))
            else:
                u = self.ver_dropout(self.de_conv(variables[self.index + 1]['z_hat']))

        psig1u = self.parsig1(u)
        psig2u = self.parsig2(u)

        varx['z_hat'] = (self.lat_dropout(varx['z_tilda']) - psig1u) * psig2u + psig1u

        if self.use_bn:
            if self.training:
                varx['z_hat_bn'] = (varx['z_hat'] - varx['mean']) / varx['std']
            else:
                assert not self.en_bn_clean.training
                varx['z_hat_bn'] = self.en_bn_clean(varx['z_hat'])
        else:
            varx['z_hat_bn'] = varx['z_hat']

        # special addition to keep the decoder from sucking needlessly
        # this has less effect than I thought it would
        # perhaps we should try and use a unique batchnorm?
        varx['z_hat_bn'] = self.de_gamma * varx['z_hat_bn'] + self.de_beta


class RLadderNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(RLadderNetwork, self).__init__()
        self.config = kwargs
        self.steps = 1

        self.num_layers, self.layer_dims, self.arglist = RLadderNetwork.gen_layer_args(**kwargs)
        self.variables = list()
        self.encoder_decoder_layers = list()

        for lidx in range(self.num_layers):
            self.variables.append(dict())
            if lidx == 0:  # the first layer
                layer = GRUED(self.layer_dims, lidx, 'first', kwargs['noise_std'], self.arglist)
            elif lidx == self.num_layers-1:  # the last layer
                layer = GRUED(self.layer_dims, lidx, 'last', kwargs['noise_std'], self.arglist)
            else:  # middle layers
                layer = GRUED(self.layer_dims, lidx, 'middle', kwargs['noise_std'], self.arglist)
            if 'batchnorm' in kwargs:
                layer.use_bn = kwargs['batchnorm']
            self.encoder_decoder_layers.append(layer)

        self.layers = nn.ModuleList(self.encoder_decoder_layers)
        self.start_epoch = 0

    @ staticmethod
    def gen_layer_args(**kwargs):
        if 'layer_dims' in kwargs:
            num_layers = len(kwargs['layer_dims'])
            layer_dims = kwargs['layer_dims']
        elif 'num_layers' in kwargs:
            assert 'in_dim' in kwargs, 'Must include \'in_dim\' when specifying \'num_layers\''
            assert 'code_dim' in kwargs, 'Must include \'code_dim\' when specifying \'num_layers\''
            num_layers = kwargs['num_layers']
            in_dim = kwargs['in_dim']
            code_dim = kwargs['code_dim']
            layer_dims = list(np.linspace(in_dim, code_dim, num_layers).round().astype(int))
        else:
            raise Exception('Must specify \'layer_dims\' or \'num_layers\'')

        arglist = list()
        for lidx in range(num_layers):
            args = {
                'kernel_size': utils.get_arg_index('kernel_size', lidx, **kwargs),
                'stride': utils.get_arg_index('stride', lidx, **kwargs),
                'padding': utils.get_arg_index('padding', lidx, **kwargs),
                'dilation': utils.get_arg_index('dilation', lidx, **kwargs)
            }
            arglist.append(args)
        return num_layers, layer_dims, arglist

    def set_lateral_weights(self, new_weight):
        for layer in self.layers:
            layer.set_lat_dropout(new_weight)

    def set_vertical_weights(self, new_weight):
        for layer in self.layers:
            layer.set_ver_dropout(new_weight)

    def set_weight(self, kind, layer_index, new_weight):
        layer = self.encoder_decoder_layers[layer_index]
        if kind is 'vertical':
            layer.set_ver_dropout(new_weight)
        elif kind is 'lateral':
            layer.set_lat_dropout(new_weight)
        else:
            raise Exception('That\'s not an option')

    def suggested_in_size(self, out_size):
        in_size = out_size
        for module in self.encoder_decoder_layers:
            in_size = module.deconvout(in_size)
        return in_size

    def set_noise_std(self, new_std):
        for module in self.layers:
            module.noise_std = new_std

    def recur(self):
        for lidx in range(self.num_layers):
            # clean pass to collect ground truth and batch statistics
            self.encoder_decoder_layers[lidx].encode_clean(self.variables)
            # noisy pass to make the architecture work for it
            self.encoder_decoder_layers[lidx].encode_noisy(self.variables)

        for lidx in reversed(range(self.num_layers)):
            # decoding pass to reconstruct input
            self.encoder_decoder_layers[lidx].decode(self.variables)

    def forward(self, **netinput):
        # setup input for network
        if 'cpu' in netinput:
            del netinput['cpu']
            self.variables[0]['x'] = netinput['x'].cpu()
        else:
            self.variables[0]['x'] = netinput['x']

        for i in range(self.steps):
            self.recur()

        output = self.make_output(netinput)
        return output

    def empty_vars(self):
        self.variables = list()
        for lidx in range(self.layers + 1):
            self.variables.append(dict())

    def make_output(self, input):
        clean = list()
        recon = list()
        for i in range(len(self.variables)):
            layer = self.variables[i]
            if i == 0:
                clean.append(layer['x'])
                recon.append(modules.downplay(layer['x'], layer['z_hat'], 20))
            else:
                clean.append(layer['z'])
                recon.append(layer['z_hat_bn'])

        output = {'clean': clean, 'recon': recon, **input}
        return output

    def save(self, apath, file='model_latest.pt'):
        save_dirs = [os.path.join(apath, file)]

        for s in save_dirs:
            torch.save(self.state_dict(), s)

    def save_model(self, path, filename):
        model = {
            'model': RLadderNetwork,
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(model, path + filename)

    def load(self, apath, file='model_latest.pt', resume=-1):
        load_from = None
        kwargs = {}
        if resume == -1:
            load_from = torch.load(os.path.join(apath, file), **kwargs)
        if load_from:
            self.load_state_dict(load_from, strict=False)

    @staticmethod
    def load_model(path, filename):
        checkpoint = torch.load(path + filename)
        model = checkpoint['model'](**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


