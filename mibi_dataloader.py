import torch
import os
import gc
from skimage import io
import numpy as np
from random import randint
from random import choice
import math
import copy
import pickle


class ImageLoader:
    def __init__(self):
        pass

    @staticmethod
    def prep_KID_img(arimg):
        x = torch.tensor(arimg)
        x = x[6:61, 29:84, :]
        x = x.float() / 255
        x = x.transpose(0, 2).transpose(1, 2).unsqueeze(0)
        return x

    @staticmethod
    def prep_KID_cmd(filename):
        filename = filename[0:-4]
        filename = filename.replace(' ', '')
        filename = filename.replace(']', '')
        filename = filename.split('[')
        timestamp = filename[0]
        if len(timestamp)==11:
            timestamp = timestamp + '0'
        timestamp = int(timestamp)
        command = filename[1]
        command = command.split(',')
        for i in range(len(command)):
            command[i] = float(command[i])
        return torch.tensor(command).float(), timestamp

    @staticmethod
    def load_pngs(**kwargs):
        directory = kwargs['folder']

        gc.disable()
        files = os.listdir(directory)
        pngs = list()
        for file in files:
            if file.endswith('.png'):
                pngs.append(file)

        if 'num' in kwargs:
            num_pngs = kwargs['num']
        else:
            num_pngs = len(pngs)

        test_img = io.imread(os.path.join(directory, pngs[0])).astype(int)
        test_img = ImageLoader.prep_KID_img(test_img)

        num_channels = test_img.shape[1]
        data_width = test_img.shape[2]
        data_height = test_img.shape[3]

        imgs = torch.zeros([num_pngs, num_channels, data_width, data_height])
        cmds = torch.zeros([num_pngs, 3])
        tsps = torch.zeros([num_pngs, 1])

        data = list()
        times = list()
        for i in range(num_pngs):
            png = pngs[i]
            command, timestamp = ImageLoader.prep_KID_cmd(png)
            path = os.path.join(directory, png)
            img = io.imread(path).astype(int)
            img = ImageLoader.prep_KID_img(img)
            data.append((img, command, timestamp))
            times.append(timestamp)
            # imgs[i, :, :, :] = img
            # cmds[i, :] = command
        indices = np.argsort(times)
        data = [data[i] for i in indices]

        for i in range(len(times)):
            imgs[i, :, :, :] = data[i][0]
            cmds[i, :] = data[i][1]
            if i > 0:
                tsps[i, :] = data[i][2] - data[i-1][2]
            else:
                tsps[i, :] = -1

        print('finished loading')
        gc.enable()
        return imgs, cmds, tsps, pngs, num_channels, data_width, data_height

    @staticmethod
    def load_folder(**kwargs):
        imglist = list()
        cmdlist = list()
        tsplist = list()

        png_data = ImageLoader.load_pngs(**kwargs)
        imglist.append(png_data[0])
        cmdlist.append(png_data[1])
        tsplist.append(png_data[2])
        nameslist = png_data[3]

        imgs = torch.cat(imglist, 0)
        cmds = torch.cat(cmdlist, 0)
        tsps = torch.cat(tsplist, 0)
        source = nameslist

        return imgs, cmds, tsps, source


class KID_Data:
    def __init__(self, **kwargs):
        image_loader = ImageLoader()

        self.flag = 'normal'
        if 'flag' in kwargs:
            self.flag = kwargs['flag']


        if 'folder' in kwargs:
            self.imgs, self.cmds, self.tsps, self.source = image_loader.load_folder(**kwargs)
        else:
            self.imgs = kwargs['imgs']
            self.cmds = kwargs['cmds']
            self.source = kwargs['source']

        self.num_points = self.imgs.size()[0]
        self.image_shape = self.imgs[0].size()  # should be [num_channels, width, height]
        self.num_channels = self.image_shape[0]
        self.channels = self.num_channels

        self.batchsize = 100
        self.crop = 32
        self.stride = 16
        self.crop_limit = self.image_shape[1] - self.crop

        if 'crop' in kwargs:
            self.set_crop(kwargs['crop'])
        if 'stride' in kwargs:
            self.set_stride(kwargs['stride'])

        if 'normalize' in kwargs:
            self.log_normalize()
        # self.global_log_normalize()

        if 'scale' in kwargs:
            for i in range(len(self.imgs)):
                self.imgs[i] = self.imgs[i] * kwargs['scale']

        print('There are ', int(self.num_points), 'samples')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

    # we know we have self.image_width and self.image_height
    # we also have self.crop
    # we need to define a stride somewhere

    # we need a way to prepare mini batches
    # each full epoch we need to go through all (self.num_points*self.vidxmax) samples

    # each epoch take a random permutation of all numbers in [0, self.num_points*self.vidxmax)
    # each minibatch take the next 'minibatchsize' set of numbers from this list
    # the image index we need is going to be float(sample_index / self.vidxmax)
    # the crop index is going to mod(sample_index, self.vidxmax)

    def set_crop(self, crop):
        self.crop = crop

    def set_stride(self, stride):
        self.stride = stride

    def get_img(self, sample_index):
        return self.imgs[sample_index]

    def get_cmd(self, sample_index):
        return self.cmds[sample_index]

    def get_tsp(self, sample_index):
        return self.tsps[sample_index]

    def prepare_epoch(self):
        # we don't want to include the first or last sample
        self.sample_queue = np.random.permutation(int(self.num_points)-2)+1

    def get_epoch_length(self):
        return len(self.imgs)-1

    def get_samples(self, sample_indices):
        samples = {
            's_i-1': torch.zeros([len(sample_indices), self.num_channels, self.crop, self.crop], dtype=torch.float32).cuda(),
            's_i':   torch.zeros([len(sample_indices), self.num_channels, self.crop, self.crop], dtype=torch.float32).cuda(),
            's_i+1': torch.zeros([len(sample_indices), self.num_channels, self.crop, self.crop], dtype=torch.float32).cuda(),
            'm_i-1': torch.zeros([len(sample_indices), 3], dtype=torch.float32).cuda(),
            'm_i':   torch.zeros([len(sample_indices), 3], dtype=torch.float32).cuda(),
            'm_i+1': torch.zeros([len(sample_indices), 3], dtype=torch.float32).cuda(),
            'dt_i-1':  torch.zeros([len(sample_indices), 1], dtype=torch.float32).cuda(),
            'dt_i':   torch.zeros([len(sample_indices), 1], dtype=torch.float32).cuda(),
            'dt_i+1':   torch.zeros([len(sample_indices), 1], dtype=torch.float32).cuda()
        }

        for i in np.arange(len(sample_indices)):
            j = sample_indices[i]
            samples['s_i-1'][i, :] = self.get_img(j-1)
            samples['s_i'][i, :] = self.get_img(j)
            samples['s_i+1'][i, :] = self.get_img(j+1)

            samples['m_i-1'][i, :] = self.get_cmd(j-1)
            samples['m_i'][i, :] = self.get_cmd(j)
            samples['m_i+1'][i, :] = self.get_cmd(j+1)

            samples['dt_i-1'][i, :] = self.get_tsp(j-1)
            samples['dt_i'][i, :] = self.get_tsp(j)
            samples['dt_i+1'][i, :] = self.get_tsp(j+1)

        return samples

    def get_next_minibatch_idxs(self, minibatch_size):
        if len(self.sample_queue) == 0:  # there is nothing left in the minibatch queue
            return None
        elif len(self.sample_queue) < minibatch_size:  # we just have to return the last of the dataset
            return None
            # minibatch_idxs = np.copy(self.sample_queue)
            # self.sample_queue = np.array([])
            # return minibatch_idxs
        else:  # we return a normal minibatch
            minibatch_idxs = np.copy(self.sample_queue[0:minibatch_size])
            self.sample_queue = self.sample_queue[minibatch_size:]
            return minibatch_idxs

    def get_next_minibatch(self, minibatch_size):
        sample_idxs = self.get_next_minibatch_idxs(minibatch_size)
        if sample_idxs is None:
            return None
        else:
            return self.get_samples(sample_idxs)

    # legacy function
    def get_batch(self, batchsize, flatten):
        # we're going to return batchsize randomly cropped imgs pulled with replacement
        # we'll also return the cmds for this batch
        sample_indices = [randint(0, self.__len__() - 1) for p in range(0, batchsize)]

        if flatten:
            sample = torch.zeros([batchsize, self.channels * self.crop * self.crop], dtype=torch.float32)
            for j in range(batchsize):
                # randcrop, a, b =
                img = self.__getitem__(sample_indices[j])
                sample[j, :] = img.reshape([torch.numel(img)])
        else:
            sample = torch.zeros([batchsize, self.channels, self.crop, self.crop], dtype=torch.float32)
            for j in range(batchsize):
                sample[j, :] = self.__getitem__(sample_indices[j])
        batch = {
            'x': torch.tensor(sample).float()
        }
        return batch

    def _log_normalize(self, x):
        nonzero_count = torch.sum(x != 0, dim=(1, 2))
        nonzero_count[nonzero_count == 0] = 1
        nonzero_sum = torch.sum(x, dim=(1, 2))
        nonzero_mean = nonzero_sum / nonzero_count.float()
        nonzero_mean[nonzero_mean == 0] = 1
        return torch.log(x / nonzero_mean.unsqueeze(-1).unsqueeze(-1) + 1)

    def log_normalize(self):
        for i in range(len(self.imgs)):
            self.imgs[i] = self._log_normalize(self.imgs[i])

    def global_log_normalize(self):
        for i in range(len(self.imgs)):
            self.imgs[i] = torch.log(self.imgs[i] + 1)

        chan_means = 0 * torch.mean(self.imgs[0], dim=(1, 2))
        for i in range(len(self.imgs)):
            chan_means += torch.mean(self.imgs[i], dim=(1, 2))
        chan_means /= len(self.imgs)

        for i in range(len(self.imgs)):
            self.imgs[i] /= chan_means.unsqueeze(-1).unsqueeze(-1)

    def pickle(self, filepath):
        pickling_on = open(filepath, 'wb')
        pickle.dump(self, pickling_on)
        pickling_on.close()

    @staticmethod
    def depickle(filepath):
        pickle_off = open(filepath, 'rb')
        dataset = pickle.load(pickle_off)
        return dataset

