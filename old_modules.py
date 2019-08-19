import torch
import torch.nn as nn
from collections import OrderedDict
import os
import numpy as np
import utils as utils
import torch.nn.functional as F


def get_batch_params(x):
    batch_size = x.shape[0]
    bessel = (batch_size - 1) / batch_size
    mean = torch.mean(x, 0)
    std = torch.sqrt(torch.var(x, 0) * bessel + 1e-05)
    return mean, std


def downplay(x, factor):
    idxs = (torch.sum(x, dim=1, keepdim=True) == 0).repeat(1,x.shape[1],1,1)
    x[idxs] = x[idxs] / factor
    return x


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        return F.dropout2d(x, self.p, True, False)

    def set_p(self, new_p):
        self.p = 1-new_p


class EncoderDecoder(nn.Module):
    def __init__(self, layer_dims, index, position, noise_std, arglist):
        super(EncoderDecoder, self).__init__()
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
            self.en_conv = nn.Conv2d(en_indim, en_outdim, bias=False, **arglist[self.index-1])
        self.en_bn_clean = nn.BatchNorm2d(en_outdim, affine=False)
        self.en_bn_noisy = nn.BatchNorm2d(en_outdim, affine=False)
        self.en_gamma = nn.Parameter(torch.rand(en_outdim, 1, 1))
        self.en_beta = nn.Parameter(torch.rand(en_outdim, 1, 1))
        self.en_nonlin = nn.ReLU()

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
        self.ver_dropout = Dropout(0.5)
        self.lat_dropout = Dropout(0.5)
        self.parsig1 = ParamSigmoid()
        self.parsig2 = ParamSigmoid()

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
            z_pre = self.en_conv(variables[self.index-1]['h'])

        # collect batch statistics
        varx['mean'], varx['std'] = get_batch_params(z_pre)

        if self.use_bn:
            varx['z'] = self.en_bn_clean(z_pre)

        # if first layer (index=0), h_(i) = z_(i)
        if self.position is 'first':
            varx['h'] = varx['z']
        else:
            # varx['h'] = self.en_nonlin(self.en_gamma * (varx['z'] + self.en_beta))  # original formulation
            varx['h'] = self.en_nonlin(self.en_gamma * varx['z'] + self.en_beta)  # I think this makes more sense

    # This function performs the noisy encoding pass of one layer of the ladder network
    def encode_noisy(self, variables):
        # print('Noisy encoder:', self.index)
        varx = variables[self.index]

        # if first layer (index=0), z_pre_tilda_(i) = x
        if self.position is 'first':
            z_pre_tilda = variables[self.index]['x']
        else:
            z_pre_tilda = self.en_conv(variables[self.index - 1]['h_tilda'])

        # we don't record the mean and std here
        if self.use_bn:
            varx['z_tilda'] = self.en_bn_noisy(z_pre_tilda) + (self.noise_std * torch.randn_like(z_pre_tilda))
        else:
            varx['z_tilda'] = z_pre_tilda + (self.noise_std * torch.randn_like(z_pre_tilda))

        # if first layer (index=0), h_tilda_(i) = z_tilda_(i)
        if self.position is 'first':
            varx['h_tilda'] = varx['z_tilda']
        else:
            # varx['h_tilda'] = self.en_nonlin(self.en_gamma * (varx['z_tilda'] + self.en_beta))  # original formulation
            varx['h_tilda'] = self.en_nonlin(self.en_gamma * varx['z_tilda'] + self.en_beta)  # ditto

    def decode(self, variables):
        # print('Decoder:', self.index)
        varx = variables[self.index]

        # if layer layer (index=L), u_(i) = de_batchnorm( h_tilda_(i) )
        if self.position is 'last':
            if self.use_bn:
                u = self.de_bn(variables[self.index]['h_tilda'])
            else:
                u = variables[self.index]['h_tilda']
        else:
            # calculate output padding
            in_shape = variables[self.index + 1]['z_hat'].shape
            w_pad = varx['z_tilda'].shape[2] - self.deconvout(in_shape[2])
            h_pad = varx['z_tilda'].shape[3] - self.deconvout(in_shape[3])
            self.de_conv.output_padding = (w_pad, h_pad)
            if self.use_bn:
                u = self.ver_dropout(self.de_bn(self.de_conv(variables[self.index + 1]['z_hat'])))
            else:
                u = self.ver_dropout(self.de_conv(variables[self.inex + 1]['z_hat']))

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


class ParamSigmoid(nn.Module):
    def __init__(self, **kwargs):
        super(ParamSigmoid, self).__init__()
        self.a1 = nn.Parameter(torch.randn(1))
        self.a2 = nn.Parameter(torch.randn(1))
        self.a3 = nn.Parameter(torch.randn(1))
        self.a4 = nn.Parameter(torch.randn(1))
        self.a5 = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.a1 * self.sigmoid(self.a2 * x + self.a3) + self.a4 * x + self.a5


class LadderNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(LadderNetwork, self).__init__()

        self.num_layers, self.layer_dims, self.arglist = LadderNetwork.gen_layer_args(**kwargs)
        self.variables = list()
        self.encoder_decoder_layers = list()

        for lidx in range(self.num_layers):
            self.variables.append(dict())
            if lidx == 0:  # the first layer
                layer = EncoderDecoder(self.layer_dims, lidx, 'first', kwargs['noise_std'], self.arglist)
            elif lidx == self.num_layers-1:  # the last layer
                layer = EncoderDecoder(self.layer_dims, lidx, 'last', kwargs['noise_std'], self.arglist)
            else:  # middle layers
                layer = EncoderDecoder(self.layer_dims, lidx, 'middle', kwargs['noise_std'], self.arglist)
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

    def forward(self, **netinput):
        # setup input for network
        if 'cpu' in netinput:
            del netinput['cpu']
            self.variables[0]['x'] = netinput['x'].cpu()
        else:
            self.variables[0]['x'] = netinput['x']

        for lidx in range(self.num_layers):
            # clean pass to collect ground truth and batch statistics
            self.encoder_decoder_layers[lidx].encode_clean(self.variables)
            # noisy pass to make the architecture work for it
            self.encoder_decoder_layers[lidx].encode_noisy(self.variables)

        for lidx in reversed(range(self.num_layers)):
            # decoding pass to reconstruct input
            self.encoder_decoder_layers[lidx].decode(self.variables)

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
                clean.append(layer['z'])
                recon.append(downplay(layer['z_hat'], 5))
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
            'model': LadderNetwork,
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


class OwlNet(nn.Module):
    def __init__(self, **kwargs):
        super(OwlNet, self).__init__()

        self.config = kwargs
        self.ladder = LadderNetwork(**kwargs)
        self.avgpool = Global_Avg_Pool()
        kwargs['code_dim'] = self.ladder.layer_dims[-1]
        self.classifier = Classifier(**kwargs)
        self.start_epoch = 0

    def set_noise_std(self, new_std):
        self.ladder.set_noise_std(new_std)

    def forward(self, **kwargs):
        owl_out = self.ladder.forward(kwargs)
        avg = self.avgpool(owl_out['recon'][-1])
        _c_ = self.classifier(avg)
        owl_out['_c_'] = _c_

        return owl_out

    def predict(self, **kwargs):
        c = kwargs['c']
        for_out = self.forward(**kwargs)
        _c_ = for_out['_c_']
        values, guesses = torch.max(_c_, 1)
        accuracy = torch.mean(1-torch.abs(guesses-c).float())
        output = {
            'guess': guesses,
            'right': accuracy
        }
        return output

    def save(self, apath, file='model_latest.pt'):
        save_dirs = [os.path.join(apath, file)]

        for s in save_dirs:
            torch.save(self.state_dict(), s)

    def save_model(self, path, filename):
        model = {
            'model': OwlNet,
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

    def load(self, apath, file='model_latest.pt', resume=-1):
        load_from = None
        kwargs = {}
        if resume == -1:
            load_from = torch.load(os.path.join(apath, file), **kwargs)
        if load_from:
            self.load_state_dict(load_from, strict=False)


class Global_Avg_Pool(nn.Module):
    def __init__(self):
        super(Global_Avg_Pool, self).__init__()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3])
        return y


class Classifier(nn.Module):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__()

        code_dim = kwargs['code_dim']
        class_dim = kwargs['class_dim']
        layers = kwargs['class_layers']

        dim_diff = float(class_dim - code_dim) / layers

        layers_dict = OrderedDict()
        for lidx in range(layers):
            fc_in_dim = int(code_dim + lidx * dim_diff)
            fc_out_dim = int(code_dim + (lidx + 1) * dim_diff)
            layers_dict['fc_' + str(lidx)] = nn.utils.weight_norm(nn.Linear(fc_in_dim, fc_out_dim), name='weight')
            # layers_dict['prelu_'+str(lidx)] = nn.PReLU()

        self.mlp = nn.Sequential(layers_dict)

    def forward(self, x):
        # if we comment out this line we allow the weights to have non-unit norm
        self.mlp.fc_0.weight.data = self.mlp.fc_0.weight.data / torch.norm(self.mlp.fc_0.weight.data, 2, 1, True)
        return self.mlp(x)


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.config = kwargs
        self.num_steps = 5
        self.in_dim = kwargs['in_dim']
        self.code_dim = kwargs['code_dim']
        self.embed_dim = kwargs['embed_dim']

        layers = 3
        dim_diff = float(self.code_dim - self.in_dim) / layers

        layers_dict = OrderedDict()
        for lidx in range(layers):
            conv_in_dim = int(self.in_dim + lidx * dim_diff)
            conv_out_dim = int(self.in_dim + (lidx + 1) * dim_diff)
            layers_dict['conv_' + str(lidx)] = nn.Conv2d(conv_in_dim, conv_out_dim, kernel_size=3, padding=1)
            layers_dict['prelu_' + str(lidx)] = nn.PReLU()

        self.conv_net = nn.Sequential(layers_dict)
        self.init_conv = nn.Conv2d(self.code_dim, self.embed_dim, kernel_size=3, padding=1)
        self.init_prelu = nn.PReLU()

        self.gru = ConvGRU(code_dim=self.code_dim, embed_dim=self.embed_dim)

        self.start_epoch = 0

    def set_num_steps(self, new_num):
        self.num_steps = new_num

    def forward(self, **kwargs):
        pidgeon_out = dict()
        y = self.init_prelu(self.conv_net(kwargs['x']))
        # we could probably initialize this more intelligently
        # h = (torch.randn(y.shape[0], self.embed_dim, y.shape[2], y.shape[3]) / 10).cuda()
        h = self.init_conv(y)
        for i in range(self.num_steps):
            h = self.gru(h, y)
        pidgeon_out['y'] = h
        return pidgeon_out

    def save(self, apath, file='model_latest.pt'):
        save_dirs = [os.path.join(apath, file)]

        for s in save_dirs:
            torch.save(self.state_dict(), s)

    def save_model(self, path, filename):
        model = {
            'model': OwlNet,
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

    def load(self, apath, file='model_latest.pt', resume=-1):
        load_from = None
        kwargs = {}
        if resume == -1:
            load_from = torch.load(os.path.join(apath, file), **kwargs)
        if load_from:
            self.load_state_dict(load_from, strict=False)


class ConvGRU(nn.Module):
    def __init__(self, **kwargs):
        super(ConvGRU, self).__init__()
        code_dim = kwargs['code_dim']
        embed_dim = kwargs['embed_dim']
        layers = kwargs['embed_layers']

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()

        self.W_h = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.W_x = nn.Conv2d(code_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.V_h = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.V_x = nn.Conv2d(code_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.U_rz = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.U_x = nn.Conv2d(code_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.b_r = nn.Parameter(torch.rand(embed_dim, 1, 1)/10)
        self.b_z = nn.Parameter(torch.rand(embed_dim, 1, 1)/10)
        self.b_h_tilda = nn.Parameter(torch.rand(embed_dim, 1, 1)/10)

    def forward(self, h, x):
        r = self.prelu(self.W_h(h) + self.W_x(x) + self.b_r)
        z = self.prelu(self.V_h(h) + self.V_x(x) + self.b_z)
        h_tilda = self.prelu(self.U_rz(r * h) + self.U_x(x) + self.b_h_tilda)
        h = (z * h) + ((1 - z) * h_tilda)
        return h

