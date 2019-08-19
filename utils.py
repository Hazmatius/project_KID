import os
import torch
import numpy as np
import math
import subprocess


def log_process(p, logfile):
    # check if file needs to be created
    f = open(logfile, 'a+')
    f.write(str(p.pid) + '\n')
    f.close()


# safe .cuda()
def scuda(obj):
    if torch.cuda.is_available():
        obj.cuda()


def tcuda(obj):
    if torch.cuda.is_available():
        obj.cuda()
    return obj


def filter_args(params, **kwargs):
    out_args = dict()
    for param in params:
        out_args[param] = kwargs[param]
    return out_args


def get_arg_default(kwarg, default, **kwargs):
    if kwarg in kwargs:
        return kwargs[kwarg]
    else:
        return default


def get_arg_defaults(defaults, **kwargs):
    for kwarg in defaults:
        if kwarg in kwargs:
            defaults[kwarg] = kwargs[kwarg]
    return defaults


def get_arg_index(kwarg, index, **kwargs):
    assert kwarg in kwargs, kwarg + ' must be provided'
    if type(kwargs[kwarg]) is int:
        return kwargs[kwarg]
    elif type(kwargs[kwarg]) is list:
        return kwargs[kwarg][index]
    else:
        raise Exception('Unhandled argument type,', type(kwargs[kwarg]), 'for keyword argument', kwarg)


def amkdir(folder):
    try:
        os.mkdir(folder)
    except:
        print(folder + ' already exists')


def global_average(z):
    return torch.mean(z, dim=[2, 3])


# we'll pass in the model, we'll pass in the weights, we'll pass in z, we'll pass in class_index
def getLatentVars(weights, z, model, epoch, var_type):
    code_dim = model.code_dim

    if var_type == 'mean':
        z = global_average(z)
        z = z.transpose(0, 1)
    elif var_type == 'total':
        z = z.transpose(0, 1)
        sz = int(z.numel() / code_dim)
        z = z.reshape([code_dim, sz])

    var = torch.std(z, dim=1) ** 2
    mean = torch.mean(z, dim=1)
    wvar = var * torch.abs(weights)

    var = var.detach().cpu()
    wvar = wvar.detach().cpu()
    weights = weights.detach().cpu()
    mean = mean.detach().cpu()

    return (var, weights, wvar, mean, epoch)


def get_latent_class_variances(model, dataset, crop_size, batch_size, epoch):
    dataset.set_crop(crop_size)
    batch = dataset.get_batch(batch_size, False)
    finput = {
        'x': batch['x'],
        'scale': 0
    }
    output = model.forward(**finput)
    _c_ = output['_c_']
    _x_ = output['_x_']
    mu = output['mu']
    logvar = output['logvar']
    z = output['z']
    # _c_, _x_, mu, logvar, z
    weights = model.classifier.mlp.fc_0.weight.data

    results = list()
    for class_index in range(model.class_dim):
        weight = weights[class_index, :]
        result = getLatentVars(weight, z, model, epoch, 'total')
        results.append(result)

    return results


import matplotlib.pyplot as plt
import imageio


def plot_class_feature_variance_curves(results, sortby, plot_shape, **kwargs):
    if ('size' in kwargs):
        fig = plt.figure(figsize=kwargs['size'])
    else:
        fig = plt.figure()
    for i in range(len(results)):
        result = results[i]

        var = result[0]
        weights = result[1]
        wvar = result[2]
        mean = result[3]
        epoch = result[4]

        if sortby == 'var':
            sort = torch.sort(var, descending=True)
        elif sortby == 'wvar':
            sort = torch.sort(wvar, descending=True)
        idx = sort[1]

        var = var / torch.max(var)
        if ('absolute' in kwargs) and (kwargs['absolute']):
            weights = torch.abs(weights)
        weights = weights / torch.max(weights)
        wvar = wvar / torch.max(wvar)

        var = var[idx].numpy()
        weights = weights[idx].numpy()
        wvar = wvar[idx].numpy()
        mean = mean[idx].numpy()

        axes = fig.add_subplot(plot_shape[0], plot_shape[1], i + 1)
        axes.plot(var, color=(0, 0, 0), label='feat. variance')
        axes.plot(weights, color=(1, 0, 0), label='class weight')
        axes.plot(wvar, color=(0, 1, 0), label='weighted var.')
        axes.legend(loc=1)
        fig.suptitle('Epoch ' + str(epoch))
    return fig


def write_var_curve_movie(path, file, sort_type, var_report):
    filename = path + file + '.gif'
    decile = int(len(var_report) / 10)
    with imageio.get_writer(filename, mode='I') as writer:
        for i in range(len(var_report)):
            fig = plot_class_feature_variance_curves(var_report[i], sort_type, (2, 1), absolute=True)
            bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
            canvas = fig.canvas
            s, (width, height) = canvas.print_to_buffer()
            image = np.fromstring(s, np.uint8).reshape((height, width, 4))
            if i % decile == 0:
                print('')
                print(i, end='')
            else:
                print('.', end='')
            writer.append_data(image)
            plt.close(fig)
    print('')
    print('done')


# for each class, return the weighted variance-sorted order of feature maps
def get_feature_order(model, dataset, crop_size, batch_size, var_type):
    model.eval()  # set network to evaluation mode
    dataset.set_crop(crop_size)  # set crop that we're going to draw from
    batch = dataset.get_batch(batch_size, False)  # get a batch (should be big to make variance of mean robust!)
    finput = {
        'x': batch['x'],
        'scale': 0
    }
    output = model.forward(**finput)
    _c_ = output['_c_']
    _x_ = output['_x_']
    mu = output['mu']
    logvar = output['logvar']
    z = output['z']
    weights = model.classifier.mlp.fc_0.weight.data  # get the classifier weights

    if var_type == 'mean':
        z = global_average(z)
        z = z.transpose(0, 1)
    elif var_type == 'total':
        z = z.transpose(0, 1)
        sz = int(z.numel() / model.code_dim)
        z = z.reshape([model.code_dim, sz])

    var = torch.std(z, dim=1) ** 2
    # now we'll treat each class separately
    class_idxs = list()
    for i in range(model.class_dim):
        w = weights[i, :]  # get the weights for class i
        wvar = var * torch.abs(w)
        wvar = wvar.detach().cpu()
        sort = torch.sort(wvar, descending=True)
        idx = sort[1]
        class_idxs.append(idx)
    return class_idxs


# x is a sample, it MAY or MAY NOT be a single image
# class_index is the class we want to to use for ordering the feature maps
# idxs is the set of all orderings
def get_feature_maps(x, class_index, idxs, model):
    model.eval()
    finput = {
        'x': x,
        'scale': 0
    }
    output = model.forward(**finput)
    _c_ = output['_c_']
    _x_ = output['_x_']
    mu = output['mu']
    logvar = output['logvar']
    z = output['z']
    # assumed x is just one sample, not a batch, so we throw away first dimension
    weights = model.classifier.mlp.fc_0.weight.data
    z = z[0, :, :, :].detach().cpu()
    if idxs != -1:
        idx = idxs[class_index]
        z = z[idx, :, :]
        w = weights[class_index, idx].clone()
    else:
        w = weights[class_index, :].clone()
    return z, w


# the intent of this is to take a single image, with it's set of feature maps,
# and the set of class weights for a given class, to linearly combine them into a single class map
def get_class_map(feature_maps, class_weights):
    z = feature_maps.cuda()  # 0-index will be feature, others will be map
    w = class_weights.clone()
    w = w.unsqueeze(1).unsqueeze(1)
    w = w.expand(w.numel(), z.shape[1], z.shape[2]).cuda()
    wz = w * z
    class_map = torch.sum(wz, dim=0)
    return class_map.detach().cpu()


def plot_feature_maps(feature_maps, class_weights, class_order, plot_shape, plot_size):
    # the maps are already sorted, we just need class_order to keep track of which feature is which
    num_plots = min([plot_shape[0] * plot_shape[1], feature_maps.numel()])
    fig = plt.figure(figsize=plot_size)
    wmaps = list()
    scales = list()
    for i in range(num_plots):
        z = feature_maps[i, :, :]
        w = class_weights[i]
        wmap = w * z
        scale = torch.std(wmap)
        wmaps.append(wmap.detach().cpu())
        scales.append(scale)
    max_scale = 4 * max(scales)
    for i in range(num_plots):
        ax = fig.add_subplot(plot_shape[0], plot_shape[1], i + 1)
        ax.grid(False)
        ax.axis('off')
        im = ax.imshow(wmaps[i], cmap='bwr', vmin=-max_scale, vmax=max_scale)
        plt.title(str(class_order[i].item()))
    return fig


def _plot_class_maps(class_maps, titles, sz, scales):
    #   %matplotlib inline
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(sz, sz))
    side = math.ceil(math.sqrt(len(class_maps)))
    for i in range(len(class_maps)):
        axes = fig.add_subplot(side, side, i + 1)
        axes.grid(False)
        axes.set_title(titles[i])
        minval = torch.abs(torch.min(class_maps[i]))
        maxval = torch.abs(torch.max(class_maps[i]))
        # scale = torch.max(minval, maxval)
        # scale = 20
        scale = scales[i]
        # scale = 4*torch.std(class_maps[i])
        img = axes.imshow(class_maps[i], cmap='bwr', vmin=-scale, vmax=scale)
        plt.axis('off')
        plt.show()
        # plt.colorbar(img, shrink=.125)

        # x = train_ds.images


def plot_class_maps(class_index, model, dataset, sz):
    x = dataset.images
    idxs = get_feature_order(model, dataset, 32, 20, 'mean')
    class_maps = list()
    titles = list()
    scales = list()
    for i in range(x.shape[0]):
        sample = torch.unsqueeze(x[i], 0)
        z, w = get_feature_maps(sample, class_index, idxs, model)
        class_map = get_class_map(z, w)
        map_min = torch.min(class_map).item()
        map_max = torch.max(class_map).item()
        class_maps.append(class_map)
        title = dataset.source[i]
        titles.append(title)
        print(title, 'min:', str(map_min), ' max:', str(map_max))
        scales.append(3)

    _plot_class_maps(class_maps, titles, sz, scales)


# def save_class_maps(model, dataset, folder):
#     # Create the appropriate directoriesf
#     folder = folder + 'class_maps/'
#     cls_dir = folder
#     amkdir(folder)
#     amkdir(cls_dir)
#
#     x = dataset.images
#     labels = dataset.labeldict  # this is a dictionary
#     print(labels)
#     inv_labels = {index.item(): label for label, index in labels.items()}
#     print(inv_labels)
#
#     # idxs = np.arange(model.code_dim)
#     # idxs = [i for i in range(model.code_dim)]
#     # Iterate over datapoints
#     for i in range(x.shape[0]):
#         sample = torch.unsqueeze(x[i], 0)
#         # Iterate over classes
#         for c in range(model.class_dim):
#             z, w = get_feature_maps(sample, c, -1, model)
#             # print(z.shape)
#             # print(w.shape)
#             class_map = get_class_map(z, w)
#             name = dataset.source[i]
#             name = os.path.splitext(name)[0]
#             name = inv_labels[c] + '_' + name
#             save_to_mat(class_map, cls_dir + name)


def plot_encodings(model, dataset, sz):
    x = dataset.images
    titles = dataset.source
    side = math.ceil(math.sqrt(x.shape[0]))
    fig = plt.figure(figsize=(sz, sz))
    for i in range(x.shape[0]):
        sample = torch.unsqueeze(x[i], 0)
        finput = {
            'x': sample,
            'temp': 0
        }
        output = model.forward(**finput)
        code = output['code']
        axes = fig.add_subplot(side, side, i + 1)
        axes.grid(False)
        axes.set_title(titles[i])
        img = axes.imshow(code[0, :, :, :].detach().cpu().transpose(0, 2).transpose(0, 1))


# def save_encoding(model, dataset, batch, folder, index, kind):
#     from scipy.misc import imsave
#
#     inimg = batch[index:index + 1, :, :, :].cuda()
#     finput = {
#         'x': inimg,
#         'temp': 0
#     }
#     output = model.forward(**finput)
#     code = output['code']
#     code = code.cpu().detach()
#     pic = code[0, :, :, :].numpy()
#     pic = np.swapaxes(pic, 0, 2)
#     pic = np.swapaxes(pic, 0, 1)
#     name = dataset.source[index]
#     name = os.path.splitext(name)[0]
#     if kind == 'png':
#         output_path = folder + name + '.png'
#         imsave(output_path, pic)
#     elif kind == 'npy':
#         output_path = folder + name
#         save_to_mat(pic.astype(np.half), output_path)
#     print('Saved encoding of ' + name)


# def save_encodings(model, dataset, folder, kind):
#     folder = folder + 'encodings/'
#     dataset.set_crop(dataset.image_width)
#     batch = dataset.images;
#     names = dataset.source;
#     amkdir(folder)
#
#     for index in range(batch.shape[0]):
#         save_encoding(model, dataset, batch, folder, index, kind)


# For saving a pytorch tensor as a .npy file
# def save_to_mat(tens, file):
#     nump = torch.tensor(tens).numpy()
#     matl = matlab.double(nump)
#     matlab.save(file + '.npy', matl)


# def save_feature_maps(model, dataset, folder):
#     # Create the appropriate directories
#     folder = folder + 'feature_maps/'
#     npy_dir = folder + 'npy/'
#     amkdir(folder)
#     amkdir(npy_dir)
#
#     # Construct and save the (unsorted) feature maps
#     x = dataset.images
#     for i in range(x.shape[0]):
#         sample = torch.unsqueeze(x[0, :, :, :], 0)
#         z, w = get_feature_maps(sample, -1, -1, model)
#         nump_z = z.numpy()
#         name = dataset.source[i]
#         name = os.path.splitext(name)[0]
#         save_to_mat(nump_z, npy_dir + name)
#         print('Saving feature maps for ' + name)


# Once we've run cateye, we should
# 1) save the feature maps for each data point
# 2) for each class, save the class map for each data point
# 3) for each data point, save its encoding

def nonzero_weight(batch_x):
    imgs = torch.sum(batch_x, dim=(1))
    imgs = imgs != 0
    size = np.prod(list(imgs.shape[1:3]))
    weights = torch.sum(imgs, dim=(1, 2)).float() / (size)
    return weights

