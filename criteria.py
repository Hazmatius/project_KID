import torch
import torch.nn as nn
import utils
import numpy as np


'''
I think we need to rethink how the combinator works
Perhaps if we add more noise, higher-level representations will work better?
Part of what we need to fix is the way that the model learns spatial representations
'''


class KIDEyeLoss(nn.Module):
    def __init__(self, **kwargs):
        super(KIDEyeLoss, self).__init__()
        self.recon_criterion = nn.MSELoss()

    def forward(self, **kwargs):
        loss = self.recon_criterion(kwargs['x'], kwargs['_x_'])
        return {'loss': loss}


class KIDMindLoss(nn.Module):
    def __init__(self, **kwargs):
        super(KIDMindLoss, self).__init__()
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.delta = kwargs['delta']
        self.for_recon_crit = nn.MSELoss()
        self.inv_recon_crit = nn.MSELoss()
        self.sen_recon_crit = nn.MSELoss()
        self.per_recon_crit = nn.MSELoss()

    def calc_sensory_loss(self, **mvars):
        s_loss1 = self.sen_recon_crit(mvars['s_i-1'], mvars['~s_i-1'])
        s_loss2 = self.sen_recon_crit(mvars['s_i'], mvars['~s_i'])
        s_loss3 = self.sen_recon_crit(mvars['s_i+1'], mvars['~s_i+1'])
        return (s_loss1 + s_loss2 + s_loss3) / 3

    def calc_perceptual_loss(self, **mvars):
        p_loss1 = self.per_recon_crit(mvars['r_i'], mvars['~r_i'])
        p_loss2 = self.per_recon_crit(mvars['r_i+1'], mvars['~r_i+1'])
        return (p_loss1 + p_loss2) / 2

    def calc_predict_loss(self, **mvars):
        pre_loss2 = self.for_recon_crit(mvars['p_i+1'], mvars['~p_i+1'])
        return pre_loss2

    def calc_posdict_loss(self, **mvars):
        pos_loss2 = self.inv_recon_crit(mvars['m_i'], mvars['~m_i'])
        return pos_loss2

    def forward(self, **kwargs):
        sen_loss = self.calc_sensory_loss(**kwargs)
        for_loss = self.calc_predict_loss(**kwargs)
        inv_loss = self.calc_posdict_loss(**kwargs)
        per_loss = self.calc_perceptual_loss(**kwargs)

        loss = self.alpha*sen_loss + self.beta*inv_loss + self.gamma*for_loss + self.delta*per_loss
        return {'loss': loss, 'inv_loss': inv_loss, 'for_loss': for_loss, 'sen_loss': sen_loss, 'per_loss': per_loss}


class OwlNetLoss(nn.Module):
    def __init__(self, **kwargs):
        super(OwlNetLoss, self).__init__()
        self.class_criterion = nn.CrossEntropyLoss()
        self.ladder_criterion = LadderNetLoss(kwargs['lambdas'])

    def forward(self, **kwargs):
        c = kwargs['c']
        _c_ = kwargs['_c_']
        recon_loss = self.ladder_criterion(**kwargs)
        class_loss = self.class_criterion(_c_, c)
        loss = recon_loss

        owlnetloss = {
            'loss': loss,
            'class': class_loss.detach().item(),
            'recon': recon_loss.detach().item()
        }
        return owlnetloss


class LadderNetLoss(nn.Module):
    def __init__(self, lambdas):
        super(LadderNetLoss, self).__init__()
        self.recon_criterion = nn.MSELoss()
        self.lambdas = lambdas

    def forward(self, **kwargs):
        # reconstruction loss
        recon_loss = 0
        for i in range(len(kwargs['clean'])):
            clean = kwargs['clean'][i]
            recon = kwargs['recon'][i]
            recon_loss += self.lambdas[i] * self.recon_criterion(recon, clean)
        laddernetloss = recon_loss
        return {'loss': laddernetloss}


class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()

    def forward(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar))
        return kl_div


class WMSELoss(nn.Module):
    def __init__(self):
        super(WMSELoss, self).__init__()

    def forward(self, input, target, weights):
        # weights = nonzero_weight(target).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mse = (input - target) ** 2
        wmseloss = (weights * ((input - target) ** 2)).mean()
        return wmseloss


class MONetLoss(nn.Module):
    def __init__(self):
        super(MONetLoss, self).__init__()

    def forward(self, **kwargs):
        beta = 0.5
        gamma = 0.1
        loss = kwargs['x_recon_loss'] + kwargs['kl_div'] + kwargs['m_recon_loss'] * gamma
        return {
            'loss': loss
        }


class CateyeLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CateyeLoss, self).__init__()

        self.class_criterion = nn.CrossEntropyLoss(reduce=False)
        self.recon_criterion = WMSELoss()  # nn.MSELoss()
        self.kldiv_criterion = KLDiv()

        self.alpha = 1  # weight for classification loss
        self.beta = 1  # weight for KL divergence
        self.gamma = 1  # weight for reconstruction loss

        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']

    def forward(self, **kwargs):
        x = kwargs['x']
        _x_ = kwargs['_x_']
        c = kwargs['c']
        _c_ = kwargs['_c_']
        mu = kwargs['mu']
        logvar = kwargs['logvar']
        # atn = kwargs['atn']

        # atn_loss = torch.mean(atn)
        weights = utils.nonzero_weight(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        class_loss = (weights * self.class_criterion(_c_, c)).mean()
        recon_loss = self.recon_criterion(_x_, x, weights)
        kldiv_loss = self.kldiv_criterion(mu, logvar)

        # cateyeloss = self.alpha*class_loss + self.gamma*recon_loss + self.beta*kldiv_loss + atn_loss
        cateyeloss = self.alpha * class_loss + self.gamma * recon_loss + self.beta * kldiv_loss
        loss_vals = {
            'loss': cateyeloss,
            'class_loss': class_loss.detach().item(),
            'recon_loss': recon_loss.detach().item(),
            'kldiv_loss': kldiv_loss.detach().item()
        }
        return loss_vals


class ManicolorLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ManicolorLoss, self).__init__()

        self.mse_loss = nn.MSELoss()

    def forward(self, **kwargs):
        x = kwargs['x'].cuda()
        _x_ = kwargs['_x_'].cuda()
        return {'loss': self.mse_loss(x, _x_)}


class PidgeonNetLoss(nn.Module):
    def __init__(self, **kwargs):
        super(PidgeonNetLoss, self).__init__()

        self.embed_criterion = EmbitterLoss()

    def forward(self, **kwargs):
        embed_loss = self.embed_criterion(**kwargs)
        return {'loss': embed_loss}


#  we need to change this so that it can take in batches
class EmbedderLoss(nn.Module):
    def __init__(self, **kwargs):
        super(EmbedderLoss, self).__init__()
        self.relu = nn.ReLU()

    @staticmethod
    def hinge_var(centroid, objs, margin):
        return torch.mean(torch.nn.functional.relu(torch.norm(centroid - objs, 2) - margin)**2)

    @staticmethod
    def hinge_dist(centroid_list, margin):
        diff_list = list()
        for i in range(len(centroid_list)):
            for j in range(len(centroid_list)):
                if i != j:
                    c_i = centroid_list[i]
                    c_j = centroid_list[j]
                    diff = torch.nn.functional.relu(margin - torch.norm(c_i - c_j, 2))**2
                    diff_list.append(diff)
        if len(diff_list) > 0:
            return torch.mean(torch.stack(diff_list))
        else:
            return torch.tensor(0).float().cuda()

    @staticmethod
    def centroid_reg(centroid_list):
        norm_list = list()
        for centroid in centroid_list:
            norm_list.append(torch.norm(centroid, 2))
        return torch.mean(torch.stack(norm_list))

    def forward(self, **kwargs):
        alpha = 1
        beta = 10
        gamma = 0.001

        y = kwargs['y']  # this is the pixel embedding layer
        c = kwargs['c']  # this is the ground truth element pixel assigments
        var_list = list()
        dis_list = list()
        reg_list = list()
        for batch_index in range(y.shape[0]):
            samp_var, samp_dis, samp_reg = self.sample_loss(y[[batch_index], :, :, :], c[batch_index, 0, :, :])
            var_list.append(samp_var)
            dis_list.append(samp_dis)
            reg_list.append(samp_reg)
        var_loss = torch.mean(torch.stack(var_list))
        dis_loss = torch.mean(torch.stack(dis_list))
        reg_loss = torch.mean(torch.stack(reg_list))
        print('.')
        print('var_loss:', (alpha*var_loss).item(), ' - dis_loss:', (beta*dis_loss).item(), '\n')
        return alpha*var_loss + beta*dis_loss

    def sample_loss(self, y, c):

        var_margin = 1  # optimize?
        dist_margin = 2  # optimize?
        labels = torch.unique(c)
        unlabel_index = (labels == -1).nonzero()
        if unlabel_index.shape[0] != 0:
            labels = torch.cat([labels[0:unlabel_index], labels[unlabel_index+1:]])

        class_var_list = list()
        centroid_list = list()

        for i in range(labels.shape[0]):
            label = labels[i]
            obj = y[:, :, c == label]
            centroid = torch.mean(obj, dim=2).unsqueeze(-1)
            var = self.hinge_var(centroid, obj, var_margin)

            class_var_list.append(var)
            centroid_list.append(centroid)

        variance_loss = torch.mean(torch.stack(class_var_list))
        distance_loss = self.hinge_dist(centroid_list, dist_margin)
        regularization = self.centroid_reg(centroid_list)

        return variance_loss, distance_loss, regularization


class EmbitterLoss(nn.Module):
    def __init__(self, **kwargs):
        super(EmbitterLoss, self).__init__()
        self.relu = nn.ReLU()

    @staticmethod
    def roll(x, n):
        return x[-n:] + x[:-n]

    @staticmethod
    def hinge_var(centroid, objs, margin):
        return torch.mean(torch.nn.functional.relu(torch.norm(centroid - objs, 2) - margin)**2)

    @staticmethod
    def hinge_dist(centroid_list, margin):
        diff_list = list()
        # centroid_list = centroid_list[np.random.permutation(int(len(centroid_list))).astype(int)]
        rolled_centroid_list = EmbitterLoss.roll(centroid_list, 1)

        centroid_tensor = torch.cat(centroid_list)
        rolled_centroid_tensor = torch.cat(rolled_centroid_list)

        diff_tensor = torch.nn.functional.relu(margin-torch.norm(centroid_tensor-rolled_centroid_tensor, 2, dim=1))**2
        if diff_tensor.shape[0]>1:
            return torch.mean(diff_tensor)
        else:
            return torch.tensor(0).float().cuda()

    @staticmethod
    def centroid_reg(centroid_list):
        norm_list = list()
        for centroid in centroid_list:
            norm_list.append(torch.norm(centroid, 2))
        return torch.mean(torch.stack(norm_list))

    def forward(self, **kwargs):
        alpha = 1
        beta = 1
        gamma = 0.001

        y = kwargs['y']  # this is the pixel embedding layer
        c = kwargs['c']  # this is the ground truth element pixel assigments
        var_list = list()
        dis_list = list()
        reg_list = list()
        for batch_index in range(y.shape[0]):
            samp_var, samp_dis, samp_reg = self.sample_loss(y[[batch_index], :, :, :], c[batch_index, 0, :, :])
            var_list.append(samp_var)
            dis_list.append(samp_dis)
            reg_list.append(samp_reg)
        var_loss = torch.mean(torch.stack(var_list))
        dis_loss = torch.mean(torch.stack(dis_list))
        reg_loss = torch.mean(torch.stack(reg_list))
        # print('.')
        # print('var_loss:', (alpha*var_loss).item(), ' - dis_loss:', (beta*dis_loss).item(), '\n')
        return alpha*var_loss + beta*dis_loss

    def sample_loss(self, y, c):

        var_margin = 1  # optimize?
        dist_margin = 3  # optimize?
        labels = torch.unique(c)
        unlabel_index = (labels == -1).nonzero()
        if unlabel_index.shape[0] != 0:
            labels = torch.cat([labels[0:unlabel_index], labels[unlabel_index+1:]])

        class_var_list = list()
        centroid_list = list()

        for i in range(labels.shape[0]):
            label = labels[i]
            obj = y[:, :, c == label]
            centroid = torch.mean(obj, dim=2).unsqueeze(-1)
            var = self.hinge_var(centroid, obj, var_margin)

            class_var_list.append(var)
            centroid_list.append(centroid)

        variance_loss = torch.mean(torch.stack(class_var_list))
        distance_loss = self.hinge_dist(centroid_list, dist_margin)
        regularization = self.centroid_reg(centroid_list)

        return variance_loss, distance_loss, regularization