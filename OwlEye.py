
# coding: utf-8

# In[5]:


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

import torch
import gc    
import os
import sys

def printmem():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_cached()
    print('Allocated:', str(allocated), '['+str(round(allocated/1000000000,3))+' GB]')
    print('   Cached:', str(cached), '['+str(round(cached/1000000000,3))+' GB]')

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
printmem()


# In[2]:


from criteria import OwlNetLoss
from helpers import Trainer
from helpers import Logger
from mibi_dataloader import MIBIData
from modules import OwlNet
import utils

printmem()


# In[3]:


# Load the data
main_dir = '/home/hazmat/Documents/mayonoise/'
train_dir = main_dir + 'data/train/'
test_dir = main_dir + 'data/test/'
modl_dir = main_dir + 'models/'
rslt_dir = main_dir + 'results/'
labels = {
    'early': torch.tensor([0]),
    'late' : torch.tensor([1])
}

train_ds = MIBIData(folder=train_dir, labels=labels, crop=32, scale=10, stride=16)
test_ds = MIBIData(folder=test_dir, labels=labels, crop=32, scale=10, stride=16)

printmem()


# In[4]:


owlnet_args = dict()
owlnet_args['kind'] = 'conv'

owlnet_args['kernel_size'] = [3,3,3,3,3]
owlnet_args['padding'] = [0,0,0,0,0]
owlnet_args['dilation'] = [1,1,1,1,1]
owlnet_args['stride'] = [1,1,1,1,1]

owlnet_args['layer_dims'] = [29, 32, 17, 9, 3]
# owlnet_args['num_layers'] = 5
# owlnet_args['in_dim'] = 29
owlnet_args['code_dim'] = 3

owlnet_args['class_dim'] = 2
owlnet_args['class_layers'] = 1

owlnet_args['noise_std'] = .2

torch.cuda.empty_cache()
owlnet = OwlNet(**owlnet_args)

owlnet.cuda()
# print(owlnet)


# In[5]:


owlnet.ladder.suggested_in_size(20)


# In[6]:


owlnet_trainer = Trainer()
owlnet_logger = Logger({'loss':(list(),list())})

# LadderNet training parameters
owlnet_train_args = dict()
owlnet_train_args['lr'] = .01
owlnet_train_args['batch_size'] = 99
owlnet_train_args['epochs'] = 1
owlnet_train_args['report'] = 5
owlnet_train_args['crop'] = 81
owlnet_train_args['clip'] = 1
owlnet_train_args['decay'] = 0
# LadderNet loss parameters
owlnet_loss_args = {}

train_ds.set_crop(owlnet_train_args['crop'])


# In[7]:


lambdas = [2**0, 2**-1, 2**-2, 2**-3, 2**-4]
owlnet.set_noise_std(0.3)
owlnet.set_lateral_weight(1)

owlnet_criterion = OwlNetLoss(lambdas, **owlnet_loss_args)
owlnet_trainer.train(owlnet, train_ds, owlnet_criterion, owlnet_logger, **owlnet_train_args)
print()
torch.cuda.empty_cache()
printmem()


# In[8]:


torch.cuda.empty_cache()
printmem()


# In[9]:


owlnet.eval()

owlnet.set_lateral_weight(1)
# %matplotlib notebook
import matplotlib.pyplot as plt
import time
print(1024/81)
train_ds.set_crop(81*1)
batch = train_ds.get_batch(1, False)
output = owlnet(**batch)

torch.cuda.empty_cache()
printmem()


# In[1]:


def print_diff2(output, model, channel, size):
    # we'going to show clean, noisy, recon, clean-recon, and noisy-recon
    fig = plt.figure(figsize=(3*size, 2*size))
    clean = output['clean'][0][0,channel,:,:].detach().cpu()
    noisy = model.ladder.variables[0]['z_tilda'][0,channel,:,:].detach().cpu()
    recon = output['recon'][0][0,channel,:,:].detach().cpu()
    # plot clean
    ax1 = plt.subplot(2,3,1)
    ax1.imshow(clean.numpy())
    plt.title('Clean')
    
    # plot noisy
    ax2 = plt.subplot(2,3,2, sharex=ax1, sharey=ax1)
    ax2.imshow(noisy.numpy())
    plt.title('Noisy')
    
    # plot clean-recon
    ax3 = plt.subplot(2,3,4, sharex=ax1, sharey=ax1)
    ax3.imshow((recon-clean).numpy())
    plt.title('Clean - Recon')
    
    # plot noisy-recon
    ax4 = plt.subplot(2,3,5, sharex=ax1, sharey=ax1)
    ax4.imshow((recon-noisy).numpy())
    plt.title('Noisy - Recon')
    
    # plot recon
    ax5 = plt.subplot(2,3,3, sharex=ax1, sharey=ax1)
    ax5.imshow(recon.numpy())
    plt.title('Recon')
    
    # plot adjustment
    ax6 = plt.subplot(2,3,6, sharex=ax1, sharey=ax1)
    noise = torch.abs(noisy-clean)
    error = torch.abs(recon-clean)
    ax6.imshow((error-noise).numpy(), cmap='bwr', vmin=-1, vmax=1)
    plt.title('Adjustment')
    
    L1_noisy_loss = torch.mean(torch.abs(noisy-clean)).item()
    L1_recon_loss = torch.mean(torch.abs(recon-clean)).item()
    L2_noisy_loss = torch.mean((noisy-clean)**2).item()
    L2_recon_loss = torch.mean((recon-clean)**2).item()
    
    print('L1 error:')
    print('  Noisy:', L1_noisy_loss)
    print('  Recon:', L1_recon_loss)
    print('L2 error:')
    print('  Noisy:', L2_noisy_loss)
    print('  Recon:', L2_recon_loss)

def print_diff(output, channel, size):
    fig = plt.figure(figsize=(3*size, size))
    ax1 = plt.subplot(1,3,1)
    recon = output['recon'][0][0,channel,:,:].detach().cpu()
    ax1.imshow(recon.numpy())
    plt.title('Reconstruction')
    
    ax2 = plt.subplot(1,3,2, sharex=ax1, sharey=ax1)
    original = output['clean'][0][0,channel,:,:].detach().cpu()
    ax2.imshow(original.numpy())
    plt.title('Original')
    
    ax3 = plt.subplot(1,3,3, sharex=ax1, sharey=ax1)
    difference = output['clean'][0][0,channel,:,:].detach().cpu() - output['recon'][0][0,channel,:,:].detach().cpu()
    ax3.imshow(difference.numpy())
    plt.title('Difference')
    fig.show()
    
    return recon, original, difference

def print_encoding(output, level, channels, size):
    print(output['recon'][level][0,:,:,:].shape)
    if isinstance(channels, list):
        if len(channels)==3:
            fig = plt.figure(figsize=(2*size, size))
            z = output['recon'][level][0,channels,:,:].transpose(0,1).transpose(1,2).detach().cpu().numpy()
            
            ax1 = plt.subplot(1,2,1)
            ax1.imshow(z)
            
            from mpl_toolkits.mplot3d import Axes3D
            data = torch.tensor(z)
            a = data[:,:,0].view(-1).numpy()
            b = data[:,:,1].view(-1).numpy()
            c = data[:,:,2].view(-1).numpy()
            ax2 = plt.subplot(1,2,2, projection='3d')
            ax2.scatter(a, b, c)
            fig.show()
            
        elif len(channels)==1:
            fig = plt.figure(figsize=(size,size))
            z = output['recon'][level][0,channels[0],:,:].detach().cpu().numpy()
            plt.imshow(z)
            fig.show()
        else:
            print('Invalid number of channels')
    else:
        fig = plt.figure(figsize=(size,size))
        z = output['recon'][level][0,channels,:,:].detach().cpu().numpy()
        plt.imshow(z)
        fig.show()
        print(z.shape)



# In[2]:


# print_encoding(output, 2, [0,1,2], 5)
# recon0, orig0, diff0 = print_diff(output,0,4)
print_diff2(output, owlnet, 0, 4)
# plt.savefig('/home/hazmat/Documents/recon1.png')
# printmem()
# torch.cuda.empty_cache()
# printmem()


# In[4]:


# print_encoding(output, 2, [0,1,2], 5)

