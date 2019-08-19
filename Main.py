# import torch
# from MIBI_Dataloader import MIBIData
# from Modules import Cateye
# from helpers import Trainer
# from helpers import Logger
# from Criteria import CateyeLoss
# import Utils
#
# # Load the data
# main_dir = '/home/hazmat/Documents/mayonoise/'
# train_dir = main_dir + 'data/train/'
# test_dir = main_dir + 'data/test/'
# modl_dir = main_dir + 'models/'
# rslt_dir = main_dir + 'results/'
# labels = {
#     'early': torch.tensor([0]),
#     'late' : torch.tensor([1])
# }
#
# train_ds = MIBIData(folder=train_dir, labels=labels, crop=32)
# test_ds = MIBIData(folder=test_dir, labels=labels, crop=32)
#
#
# # Cateye model parameters
# cateye_model_args = {}
# cateye_model_args['kernel'] = 5
# cateye_model_args['in_dim'] = 29
# cateye_model_args['code_dim'] = 16
# cateye_model_args['class_dim'] = 2
# cateye_model_args['encoder_layers'] = 2
# cateye_model_args['decoder_layers'] = 2
# cateye_model_args['attention_layers'] = 2
# cateye_model_args['class_layers'] = 1
#
# # Manicolor model parameters
# manicolor_model_args = {}
# manicolor_model_args['chan_dims'] = [29, 29+3, 10, 3]
# manicolor_model_args['kind'] = 'sigmoid'
#
# cateye = Cateye(**cateye_model_args)
# cateye.cuda()
# # print(cateye)
#
# print(cateye)
#
#
# cateye_trainer = Trainer()
# cateye_logger = Logger({'loss':(list(),list())})
#
# variational = True
# # Cateye training parameters
# cateye_train_args = {}
# cateye_train_args['lr'] = 0.001
# cateye_train_args['batch_size'] = 100
# cateye_train_args['epochs'] = 5
# cateye_train_args['report'] = 50
# cateye_train_args['crop'] = 64
# cateye_train_args['clip'] = 1
# cateye_train_args['scale'] = 1
# cateye_train_args['decay'] = 0
# # Cateye loss parameters
# cateye_loss_args = {}
# cateye_loss_args['alpha'] = 1 # Cross-entropy classification loss
# cateye_loss_args['beta'] = 1 # KL-divergence
# cateye_loss_args['gamma'] = 0.01 # MSE reconstruction loss
# if not variational:
#   cateye_train_args['scale'] = 0
#   cateye_loss_args['beta'] = 0
#
# # Manicolor training parameters
# manicolor_train_args = {}
# manicolor_train_args['lr'] = 0.01
# manicolor_train_args['batch_size'] = 50
# manicolor_train_args['epochs'] = 500
# manicolor_train_args['report'] = 50
# manicolor_train_args['crop'] = 32
# manicolor_train_args['clip'] = 1
# manicolor_train_args['decay'] = 0
# manicolor_train_args['temp'] = 0.01
# # Manicolor loss parameters
# manicolor_loss_args = {}
#
# cateye_criterion = CateyeLoss(**cateye_loss_args)
# cateye_trainer.train(cateye, train_ds, cateye_criterion, cateye_logger, **cateye_train_args)
#
# # Utils.plot_class_maps(0, cateye, train_ds, 20)
# # manicolor_criterion = ManicolorLoss(**manicolor_loss_args)
# # manicolor_trainer.train(manicolor, train_ds, manicolor_criterion, manicolor_logger, **manicolor_train_args)
#
# # Utils.save_feature_maps(cateye, train_ds, main_dir)
# # Utils.save_feature_maps(cateye, test_ds, main_dir)
# # Utils.save_class_maps(cateye, train_ds, main_dir)
# # Utils.save_class_maps(cateye, test_ds, main_dir)