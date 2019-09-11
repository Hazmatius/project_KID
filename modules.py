import torch
import torch.nn as nn
from collections import OrderedDict
import os
import numpy as np
import utils as utils
import torch.nn.functional as F


def downsample(x):
    x[:, :, 1::2, ::2] = x[:, :, ::2, ::2]
    x[:, :, ::2, 1::2] = x[:, :, ::2, ::2]
    x[:, :, 1::2, 1::2] = x[:, :, ::2, ::2]
    # x[:, :, ::2+1, ::2+1] = 0
    return x


def get_batch_params(x):
    batch_size = x.shape[0]
    bessel = (batch_size - 1) / batch_size
    mean = torch.mean(x, 0)
    std = torch.sqrt(torch.var(x, 0) * bessel + 1e-05)
    return mean, std


# x is the 'source' of downplaying, y is the 'target' of downplaying
def downplay(x, y, factor):
    idxs = (torch.sum(x, dim=1, keepdim=True) == 0).repeat(1,x.shape[1],1,1)
    y[idxs] = y[idxs] / factor
    return y


# sensory model
# responsible for compressing an image of a finger into a 3d vector
class SensoryModel(nn.Module):
    def __init__(self, **kwargs):
        super(SensoryModel, self).__init__()

        self.config = kwargs
        self.start_epoch = 0
        self.shape = None
        self.noise_level = 0

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(8, affine=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=0, stride=2, dilation=1)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=0, stride=2, dilation=1)
        self.bn3 = nn.BatchNorm2d(32, affine=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=0, stride=2, dilation=1)
        self.bn4 = nn.BatchNorm2d(32, affine=False)

        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, padding=0, stride=2, dilation=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, padding=0, stride=2, dilation=1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, padding=0, stride=2, dilation=1)
        self.deconv1 = nn.ConvTranspose2d(8, 3, kernel_size=3, padding=0, stride=1, dilation=1)

        self.fc1 = nn.Linear(512, 384)
        self.bn5 = nn.BatchNorm1d(384, affine=False)
        self.fc2 = nn.Linear(384, 258)
        self.bn6 = nn.BatchNorm1d(258, affine=False)
        self.fc3 = nn.Linear(258, 131)
        self.bn7 = nn.BatchNorm1d(131, affine=False)
        self.fc4 = nn.Linear(131, 3)
        self.bn8 = nn.BatchNorm1d(3, affine=False)

        self.defc4 = nn.Linear(3, 131)
        self.defc3 = nn.Linear(131, 258)
        self.defc2 = nn.Linear(258, 384)
        self.defc1 = nn.Linear(384, 512)

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def conv_encode(self, x):
        noise = (2*torch.rand_like(x)-1) * self.noise_level
        x = x + noise
        y = self.prelu(self.bn1(self.conv1(x)))
        y = self.prelu(self.bn2(self.conv2(y)))
        y = self.prelu(self.bn3(self.conv3(y)))
        y = self.prelu(self.bn4(self.conv4(y)))
        return y

    def conv_decode(self, y):
        _x = self.prelu(self.deconv4(y))
        _x = self.prelu(self.deconv3(_x))
        _x = self.prelu(self.deconv2(_x))
        _x = self.prelu(self.deconv1(_x))
        return _x

    def fc_encode(self, x):
        self.shape = x.shape
        y = x.flatten(1)
        y = self.tanh(self.bn5(self.fc1(y)))
        y = self.tanh(self.bn6(self.fc2(y)))
        y = self.tanh(self.bn7(self.fc3(y)))
        y = self.tanh(self.bn8(self.fc4(y)))
        return y

    def fc_decode(self, y):
        x = self.prelu(self.defc4(y))
        x = self.prelu(self.defc3(x))
        x = self.prelu(self.defc2(x))
        x = self.prelu(self.defc1(x))
        x = x.view(self.shape)
        return x

    def encode(self, x):
        return self.fc_encode(self.conv_encode(x))

    def decode(self, y):
        # y = y + (2*torch.rand_like(y)-1)*self.noise_level
        return self.conv_decode(self.fc_decode(y))

    def forward(self, s):
        r = self.encode(s)
        _s_ = self.decode(r)
        return r, _s_


# perceptual model
class PerceptualModel(nn.Module):
    def __init__(self, **kwargs):
        super(PerceptualModel, self).__init__()

        self.config = kwargs
        self.start_epoch = 0
        self.noise_level = 0

        self.fc1 = nn.Linear(9, 18)
        self.bn1 = nn.BatchNorm1d(18, affine=False)
        self.fc2 = nn.Linear(18, 36)
        self.bn2 = nn.BatchNorm1d(36, affine=False)
        self.fc3 = nn.Linear(36, 72)
        self.bn3 = nn.BatchNorm1d(72, affine=False)
        self.fc4 = nn.Linear(72, 9)
        self.bn4 = nn.BatchNorm1d(9, affine=False)

        self.defc4 = nn.Linear(9, 72)
        self.debn4 = nn.BatchNorm1d(72, affine=False)
        self.defc3 = nn.Linear(72, 36)
        self.debn3 = nn.BatchNorm1d(36, affine=False)
        self.defc2 = nn.Linear(36, 18)
        self.debn2 = nn.BatchNorm1d(18, affine=False)
        self.defc1 = nn.Linear(18, 3)

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def encode(self, r_0, r_1, m_0):
        # r_0 is previous state, r_1 is current state, m_0 is previous action
        p = torch.cat([r_0, r_1, m_0], 1)
        noise = (2*torch.rand_like(p)-1) * self.noise_level
        p = p + noise
        p = self.prelu(self.bn1(self.fc1(p)))
        p = self.prelu(self.bn2(self.fc2(p)))
        p = self.prelu(self.bn3(self.fc3(p)))
        p = self.tanh(self.bn4(self.fc4(p)))
        return p

    def decode(self, p):
        # p = p + (2*torch.rand_like(p)-1) * self.noise_level
        _r_1 = self.prelu(self.debn4(self.defc4(p)))
        _r_1 = self.prelu(self.debn3(self.defc3(_r_1)))
        _r_1 = self.prelu(self.debn2(self.defc2(_r_1)))
        _r_1 = self.defc1(_r_1)
        return _r_1

    def forward(self, r_0, r_1, m_0):
        p = self.encode(r_0, r_1, m_0)
        _r_1 = self.decode(p)
        return p, _r_1


# forward kinematics model
# takes in a perceptual state and an action, returning a predicted next state
class ForwardKinematicsModel(nn.Module):
    def __init__(self, **kwargs):
        super(ForwardKinematicsModel, self).__init__()

        self.config = kwargs
        self.start_epoch = 0

        self.fc1 = nn.Linear(9+3, 18)
        self.bn1 = nn.BatchNorm1d(18, affine=False)
        self.fc2 = nn.Linear(18, 18)
        self.bn2 = nn.BatchNorm1d(18, affine=False)
        self.fc3 = nn.Linear(18, 9)
        self.bn3 = nn.BatchNorm1d(9, affine=False)
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, p_0, m_0):
        x = torch.cat([p_0, m_0], 1)
        _p_1 = self.prelu(self.bn1(self.fc1(x)))
        _p_1 = self.prelu(self.bn2(self.fc2(_p_1)))
        _p_1 = self.tanh(self.bn3(self.fc3(_p_1)))
        return _p_1


# inverse kinematics model
# takes in a current perceptual state and a previous perceptual state,
# and predicts the action that caused the transition
class InverseKinematicsModel(nn.Module):
    def __init__(self, **kwargs):
        super(InverseKinematicsModel, self).__init__()

        self.config = kwargs
        self.start_epoch = 0
        self.indices = ['i-1', 'i', 'i+1']

        self.fc1 = nn.Linear(9+9+1, 36)
        self.fc_skip = nn.Linear(9+9+1, 3)
        self.bn1 = nn.BatchNorm1d(36, affine=False)
        self.fc2 = nn.Linear(36, 30)
        self.bn2 = nn.BatchNorm1d(30, affine=False)
        self.fc3 = nn.Linear(30, 20)
        self.bn3 = nn.BatchNorm1d(20, affine=False)
        self.fc4 = nn.Linear(20, 10)
        self.bn4 = nn.BatchNorm1d(10, affine=False)
        self.fc5 = nn.Linear(10, 3)
        self.bn5 = nn.BatchNorm1d(3, affine=False)
        self.fc6 = nn.Linear(3, 3)

        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, p_0, p_1, dt):
        x = torch.cat([p_0, p_1, dt], 1)
        _m_0 = self.prelu(self.bn1(self.fc1(x)))
        _m_0 = self.prelu(self.bn2(self.fc2(_m_0)))
        _m_0 = self.prelu(self.bn3(self.fc3(_m_0)))
        _m_0 = self.prelu(self.bn4(self.fc4(_m_0)))
        _m_0 = self.tanh(self.bn5(self.fc5(_m_0) + self.fc_skip(x)))
        _m_0 = self.fc6(_m_0)
        return _m_0


class Mind_of_KID(nn.Module):
    def __init__(self, **kwargs):
        super(Mind_of_KID, self).__init__()

        self.config = kwargs
        self.start_epoch = 0
        self.indices = ['i-1', 'i', 'i+1']

        self.sensory_model = SensoryModel(**kwargs)
        self.perceptual_model = PerceptualModel(**kwargs)
        self.forward_kinematics_model = ForwardKinematicsModel(**kwargs)
        self.inverse_kinematics_model = InverseKinematicsModel(**kwargs)

    def set_noise_level(self, noise_level):
        self.sensory_model.noise_level = noise_level
        self.perceptual_model.noise_level = noise_level

    def index(self, var, i):
        return var + '_' + self.indices[i+1]

    def sensory_encode(self, i, **mvars):
        s = mvars[self.index('s', i)]
        r, _s = self.sensory_model.forward(s)
        mvars[self.index('r', i)] = r
        mvars[self.index('~s', i)] = _s
        return mvars

    def perceptual_encode(self, i, **mvars):
        r_0 = mvars[self.index('r', i-1)]
        r_1 = mvars[self.index('r', i)]
        m_0 = mvars[self.index('m', i-1)]
        p, _r_1 = self.perceptual_model.forward(r_0, r_1, m_0)
        mvars[self.index('p', i)] = p
        mvars[self.index('~r', i)] = _r_1
        return mvars

    def predict(self, i, **mvars):
        p_0 = mvars[self.index('p', i-1)]
        m_0 = mvars[self.index('m', i-1)]
        _p_1 = self.forward_kinematics_model.forward(p_0, m_0)
        mvars[self.index('~p', i)] = _p_1
        return mvars

    def postdict(self, i, **mvars):
        dt = mvars[self.index('dt', i+1)]
        p_0 = mvars[self.index('p', i)]
        p_1 = mvars[self.index('p', i+1)]
        _m_0 = self.inverse_kinematics_model.forward(p_0, p_1, dt)
        mvars[self.index('~m', i)] = _m_0
        return mvars

    def forward(self, **mvars):
        # encode sensory states
        mvars = self.sensory_encode(-1, **mvars)
        mvars = self.sensory_encode(0, **mvars)
        mvars = self.sensory_encode(1, **mvars)

        mvars = self.perceptual_encode(0, **mvars)
        mvars = self.perceptual_encode(1, **mvars)

        mvars = self.predict(1, **mvars)

        mvars = self.postdict(0, **mvars)
        return mvars

    def save(self, apath, file='model_latest.pt'):
        save_dirs = [os.path.join(apath, file)]

        for s in save_dirs:
            torch.save(self.state_dict(), s)

    def save_model(self, path, filename):
        model = {
            'model': Mind_of_KID,
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
        if torch.cuda.is_available():
            checkpoint = torch.load(path + filename, map_location='cuda')
        else:
            checkpoint = torch.load(path + filename, map_location='cpu')
        model = checkpoint['model'](**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


# class KID_Mover(nn.Module):
#     def __init__(self, **kwargs):
#         super(KID_Mover, self).__init__()
#
#         self.config = kwargs
#         self.start_epoch = 0
#
#


class KID_Eye(nn.Module):
    def __init__(self, **kwargs):
        super(KID_Eye, self).__init__()

        self.config = kwargs
        self.start_epoch = 0

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=0, stride=2, dilation=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=0, stride=2, dilation=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=0, stride=2, dilation=1)

        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, padding=0, stride=2, dilation=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, padding=0, stride=2, dilation=1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=5, padding=0, stride=2, dilation=1)
        self.deconv1 = nn.ConvTranspose2d(8, 3, kernel_size=3, padding=0, stride=1, dilation=1)
        self.nonlin = nn.PReLU()

    def forward(self, **kwargs):
        x = kwargs['s_t']
        z = self.nonlin(self.conv1(x))
        z = self.nonlin(self.conv2(z))
        z = self.nonlin(self.conv3(z))
        z = self.nonlin(self.conv4(z))

        _x_ = self.nonlin(self.deconv4(z))
        _x_ = self.nonlin(self.deconv3(_x_))
        _x_ = self.nonlin(self.deconv2(_x_))
        _x_ = self.nonlin(self.deconv1(_x_))

        output = {
            'x': x,
            'z': z,
            '_x_': _x_
        }

        return output

    def save(self, apath, file='model_latest.pt'):
        save_dirs = [os.path.join(apath, file)]

        for s in save_dirs:
            torch.save(self.state_dict(), s)

    def save_model(self, path, filename):
        model = {
            'model': KID_Eye,
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

