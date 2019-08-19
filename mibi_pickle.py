import sys
import json
import os
import copy
import mibi_dataloader

json_file = sys.argv[1]
with open(json_file) as json_data:
    hyperconfig = json.load(json_data)

dataset_params = hyperconfig['dataset_params']
output_params = hyperconfig['output_params']

train_folder = dataset_params['data_dir'] + dataset_params['train_dir']
test_folder = dataset_params['data_dir'] + dataset_params['test_dir']
train_ds_params = {
    'folder': train_folder,
    'crop': dataset_params['crop'],
    'scale': dataset_params['scale'],
    'stride': dataset_params['stride']
}
test_ds_params = {
    'folder': test_folder,
    'crop': dataset_params['crop'],
    'scale': dataset_params['scale'],
    'stride': dataset_params['stride']
}
if 'labels' in dataset_params:
    train_ds_params['labels'] = dataset_params['labels']
    test_ds_params['labels'] = dataset_params['labels']

train_ds = mibi_dataloader.MIBIData(**copy.deepcopy(train_ds_params))
test_ds = mibi_dataloader.MIBIData(**copy.deepcopy(test_ds_params))
train_ds_path = output_params['hyper_dir'] + 'datasets/train_ds.pickle'
test_ds_path = output_params['hyper_dir'] + 'datasets/test_ds.pickle'

if not os.path.exists(output_params['hyper_dir'] + 'datasets/'):
    os.makedirs(output_params['hyper_dir'] + 'datasets/')
train_ds.pickle(train_ds_path)
test_ds.pickle(test_ds_path)