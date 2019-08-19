import torch
import time
import json
import copy
import random
import string
import os
import traceback
import numpy as np
import subprocess
import shutil
import sys
import modules


import utils as utils
import mibi_dataloader
import modules
import criteria


# helper function for dict_factor
def dict_prod(key, vals, dict_list):
    dict_list_prod = []
    for val in vals:
        dict_list_copy = copy.deepcopy(dict_list)
        for dictionary in dict_list_copy:
            dictionary[key] = val
            dict_list_prod.append(dictionary)
    return dict_list_prod


# dict_factor takes in a dictionary with list of values as key-values
# the output dictionary is a cartesian product of the values in the input
def dict_factor(dictionary):
    dict_list = [copy.copy(dictionary)]
    for key in dictionary:
        vals = dictionary[key]
        dict_list = dict_prod(key, vals, dict_list)
    return dict_list


def printmem():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_cached()
    print(
        'Allocated:',
        str(allocated),
        '['+str(round(allocated*1e-9, 3)) + ' GB]',
        '|',
        'Cached:',
        str(cached),
        '['+str(round(cached*1e-9, 3)) + ' GB]'
    )


'''
HyperSearcher is a class used to organize hyperparameter search.
It takes in a path to a json file that specifies the parameters to search over

'''
class HyperSearcher:
    def __init__(self):
        pass

    # takes in a path to a json file specifying the hyperparameters to search over
    def run_hypersearch(self, json_file):
        with open(json_file) as json_data:
            hypersearch_config = json.load(json_data)
        # pickle our datasets so they're faster to load
        hypersearch_config = HyperSearcher.pickle_datasets(hypersearch_config)

        # generate all combinations of searched hyperparameters

        # check if there is already an all_trials.json
        print(hypersearch_config['output_params']['hyper_dir'] + 'all_trials.json')
        if os.path.isfile(hypersearch_config['output_params']['hyper_dir'] + 'all_trials.json'):
            print('Old trials config json found, proceeding with original file...')
            with open(hypersearch_config['output_params']['hyper_dir'] + 'all_trials.json') as all_trials_json:
                trials_dict = json.load(all_trials_json)
                hyper_configs = list()
                for key in trials_dict.keys():
                    hyper_configs.append(trials_dict[key])
        else:
            hyper_configs, trials_dict = HyperSearcher.gen_trial_params(hypersearch_config)

        num_trials = hypersearch_config['search_params']['trials']
        # Test hyperparams for errors before running
        hyperpass = HyperSearcher.check_hyperpoints(hypersearch_config, hyper_configs)
        if hyperpass:
            # run all hyperpoints
            HyperSearcher.train_hyperpoints(trials_dict, hypersearch_config, hyper_configs, num_trials)
        else:
            print('\n........some parameters were invalid, terminating hypersearch........\n')

    @staticmethod
    def train_hyperpoints(trials_dict, hypersearch_config, hyper_configs, num_trials):
        all_trials_json = json.dumps(trials_dict, indent=4)
        f = open(hypersearch_config['output_params']['hyper_dir'] + 'all_trials.json', 'w')
        f.write(all_trials_json)
        f.close()

        print('\nAll parameters combinations are valid, proceeding with hypersearch.\n')
        count = 0
        for hyper_config in hyper_configs:
            count += 1
            print(hyper_config['trial_name'] + ': ' + str(count) + '/' + str(len(hyper_configs)))
            HyperSearcher.run_hyper_config(hyper_config, hypersearch_config, num_trials)

    '''
    Runs a specific set of hyperparameters for the specified number of trials
    Parameters:
        hyper_config (dict): dictionary of hyperparameters
        hypersearch_config (dict): parent dictionary that contains general information
        num_trials (int): the number of trials to run with these hyperparameters
    '''
    @staticmethod
    def run_hyper_config(hyper_config, hypersearch_config, num_trials):
        hyper_dir = hyper_config['output_params']['results_dir'] + hyper_config['trial_name'] + '/'
        if not os.path.exists(hyper_dir):
            os.makedirs(hyper_dir)
        hyper_json = json.dumps(hyper_config, indent=4)
        f = open(hyper_dir + 'hyperpoint.json', 'w')
        f.write(hyper_json)
        f.close()
        for trial in range(num_trials):
            print('> trial_', str(trial))
            p = subprocess.Popen(['python', 'run_trial.py', 'train', json.dumps(hyper_config), str(trial)])
            utils.log_process(p, hypersearch_config['output_params']['hyper_dir'] + 'process_log.txt')
            p.communicate()

    @staticmethod
    def check_hyperpoints(hypersearch_config, hyper_configs):
        hyperpass = True
        count = 0
        print('Testing...')
        train_ds_path = hypersearch_config['dataset_params']['train_ds_path']
        for hyper_config in hyper_configs:
            count += 1
            print('\rTesting ' + str(count) + '/' + str(len(hyper_configs)) + ': ' + hyper_config['trial_name'], end='')
            p = subprocess.Popen(['python', 'run_trial.py', 'check', json.dumps(hyper_config), train_ds_path])
            utils.log_process(p, hypersearch_config['output_params']['hyper_dir'] + 'process_log.txt')
            p.communicate()
            return_code = p.returncode
            hyperpass = hyperpass and return_code == 0
        return hyperpass

    '''
    this function is lying
    '''
    @staticmethod
    def pickle_datasets(hyperconfig):
        output_params = hyperconfig['output_params']

        train_ds_path = output_params['hyper_dir'] + 'datasets/train_ds.pickle'
        test_ds_path = output_params['hyper_dir'] + 'datasets/test_ds.pickle'

        hyperconfig['dataset_params']['train_ds_path'] = train_ds_path
        hyperconfig['dataset_params']['test_ds_path'] = test_ds_path

        return hyperconfig

    @staticmethod
    def gen_trial_params(hyperconfig):
        dataset_params = hyperconfig['dataset_params']
        output_params = hyperconfig['output_params']
        model_params = hyperconfig['model_params']
        train_params = hyperconfig['train_params']
        loss_params = hyperconfig['loss_params']

        all_model_params = dict_factor(model_params)
        all_train_params = dict_factor(train_params)
        trial_params = {
            'model_params': all_model_params,
            'train_params': all_train_params,
        }

        # construct all combinations of trial parameters
        trials_dict = dict()
        all_trial_params = dict_factor(trial_params)
        all_trial_foldernames = list()
        for trial_params in all_trial_params:
            nfn = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=8))
            while nfn in all_trial_foldernames:
                nfn = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=8))
            all_trial_foldernames.append(nfn)
            trial_params['trial_name'] = nfn
            trial_params['dataset_params'] = dataset_params
            trial_params['output_params'] = output_params
            trial_params['loss_params'] = loss_params
            trials_dict[nfn] = trial_params

        return all_trial_params, trials_dict


class Trial:
    def __init__(self, config, trial_num, model_class):
        self.done = False
        self.config = config
        self.trail_num = trial_num
        self.training_time = -1
        # using the trial dictionary, we should create the results folder
        self.hyper_dir = self.config['output_params']['results_dir'] + self.config['trial_name']
        self.trial_dir = self.hyper_dir + '/' + 'trial_' + str(trial_num) + '/'

        if os.path.exists(self.trial_dir):
            self.done = self.check_if_done()
            if not self.done:
                shutil.rmtree(self.trial_dir)

        if not os.path.exists(self.trial_dir):
            os.makedirs(self.trial_dir)

        self.model = model_class(**copy.deepcopy(self.config['model_params']))
        self.model.cuda()
        self.trainer = Trainer()
        self.logger = Logger({'loss': (list(), list())})
        self.test_logger = Logger({'loss': (list(), list())})
        self.criterion = criteria.LadderNetLoss(**copy.deepcopy(self.config['loss_params']))

    @staticmethod
    def error_check(config, dataset_path, model_class):
        dataset = mibi_dataloader.MIBIData.depickle(dataset_path)
        param_pass = True
        # test that model params were valid
        try:
            model = model_class(**copy.deepcopy(config['model_params']))
            model.cuda()
        except Exception as e:
            tb = traceback.format_exc()
            print('ERROR INITIALIZING MODEL:')
            print(config['model_params'])
            print(tb)
            param_pass = False
        trainer = Trainer()
        try:
            criterion = criteria.LadderNetLoss(**copy.deepcopy(config['loss_params']))
        except Exception as e:
            tb = traceback.format_exc()
            print('ERROR INITIALIZING CRITERION:')
            print(config['loss_params'])
            print(tb)
            param_pass = False
        if param_pass:
            try:
                trainer.error_check(model, dataset, criterion, **config['train_params'])
            except Exception as e:
                tb = traceback.format_exc()
                print('ERROR DURING TRAINING:')
                print(config['train_params'])
                print(tb)
                param_pass = False
        return param_pass

    def check_if_done(self):
        model_check = os.path.isfile(self.trial_dir + 'model')
        summary_check = os.path.isfile(self.trial_dir + 'summary.json')
        trainloss_check = os.path.isfile(self.trial_dir + 'train_loss')
        return model_check and summary_check and trainloss_check

    def train(self):
        if not self.done:
            train_ds = mibi_dataloader.MIBIData.depickle(self.config['dataset_params']['train_ds_path'])
            self.training_time = self.trainer.train(
                self.model,
                train_ds,
                self.criterion,
                self.logger,
                **self.config['train_params']
            )

    def test(self):
        if not self.done:
            test_ds = mibi_dataloader.MIBIData.depickle(self.config['dataset_params']['test_ds_path'])
            Trainer.test(
                self.model,
                test_ds,
                self.criterion,
                self.test_logger,
                **self.config['train_params']
            )

    def save(self):
        self.model.save_model(self.trial_dir, 'model')
        self.logger.save_loss_log(self.trial_dir, 'train_loss')
        training_loss = self.logger.get_final_avg_loss(amount=100)
        test_loss = self.test_logger.get_final_avg_loss()
        generalization_error = test_loss - training_loss
        summary = {
            'training_time': str(self.training_time),
            'trianing_loss': training_loss,
            'test_loss': test_loss,
            'generalization_error': generalization_error
        }
        summary_json = json.dumps(summary, indent=4)
        f = open(self.trial_dir + 'summary.json', 'w')
        f.write(summary_json)
        f.close()


# list_vars should be a dictionary initialized with lists
# logging functions take in arguments like epoch, {'var': value}
class Logger:
    def __init__(self, list_vars):
        self.list_vars = dict()
        for var in list_vars:
            self.list_vars[var] = (list(), list())

    def log(self, epoch, **kwargs):
        for var in self.list_vars:
            value = kwargs[var]
            if torch.numel(value) == 1:
                value = value.detach().cpu().item()
            self.list_vars[var][0].append(epoch)
            self.list_vars[var][1].append(value)

    def save_loss_log(self, folder, file):
        np.savetxt(folder+file, self.list_vars['loss'][1], delimiter=',')

    def get_final_avg_loss(self, **kwargs):
        if 'amount' in kwargs:
            amount = kwargs['amount']
            try:
                final_avg_loss = np.average(self.list_vars['loss'][1][-amount::])
            except Exception as e:
                print('ACHTUNG!', e)
                final_avg_loss = np.average(self.list_vars['loss'][1])
        else:
            final_avg_loss = np.average(self.list_vars['loss'][1])
        return final_avg_loss


# The purpose of this class is to encapsulate functionality related to training a model,
# with support for making error plots, saving optimizer parameters, etc.
class Trainer:
    def __init__(self, **kwargs):
        self.optimizer = None
        self.lr_scheduler = None
        self.epsilon = 0
        self.loss_sum = 0

    def clip_gradient(self, model, clip):
        if clip is None:
            return
        totalnorm = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad.data = p.grad.data.clamp(-clip, clip)

    def print_dict(self, dict):
        printout = ''
        for key in dict.keys():
            var = dict[key]
            if isinstance(var, torch.Tensor):
                printout = printout + key + ': ' + str(round(var.detach().item(), 5)) + ', '
            else:
                printout = printout + key + ': ' + str(round(var, 5)) + ', '
        return printout

    def error_check(self, model, dataset, criterion, **kwargs):
        model.train()
        defaults = {'lr': 0.01, 'clip': None, 'decay': 0, 'crop': 32, 'batch_size': 100}
        kwargs = utils.get_arg_defaults(defaults, **kwargs)

        dataset.set_crop(kwargs['crop'])
        dataset.prepare_epoch()
        batch_vars = dataset.get_next_minibatch(kwargs['batch_size'])

        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['decay'])

        model_vars = model.forward(**{**batch_vars, **kwargs})
        error_vars = criterion(**{**batch_vars, **model_vars, **kwargs})
        loss = error_vars['loss']

        optimizer.zero_grad()
        loss.backward()
        if kwargs['clip'] is not None:
            self.clip_gradient(model, kwargs['clip'])
        optimizer.step()

    def train(self, model, train_set, criterion, logger, home_dir, **kwargs):
        model.train()
        defaults = {'lr': 0.01, 'batch_size': 100, 'epochs': 10, 'report': 5, 'crop': 32, 'clip': None, 'decay': 0,
                    'epoch_frac': 1, 'restart': -1}
        kwargs = utils.get_arg_defaults(defaults, **kwargs)
        # if there is no optimizer or we want to use a new learning rate, instantiate Adam optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['decay'])
        if self.optimizer is not None and ('continue' in kwargs and not kwargs['continue']):
            self.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['decay'])

        num_minibatches = int(train_set.get_epoch_length() / float(kwargs['batch_size']))
        train_set.set_crop(kwargs['crop'])
        t = time.time()

        training_dir = home_dir + time.strftime('%Y%b%d_%H-%M-%S') + '/'
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        restart_countdown = 0
        for epoch in range(model.start_epoch, model.start_epoch + kwargs['epochs']):
            restart_countdown += 1
            if restart_countdown == kwargs['restart']:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['decay'])
                restart_countdown = 0
            train_set.prepare_epoch()
            self.loss_sum = minibatch_number = frac = batch_vars = 0
            while (batch_vars is not None) and (frac < kwargs['epoch_frac']):
                frac = kwargs['batch_size'] * minibatch_number / float(train_set.get_epoch_length())
                batch_vars = train_set.get_next_minibatch(kwargs['batch_size'])
                if batch_vars is not None:
                    model_vars = model.forward(**batch_vars)
                    error_vars = criterion(**{**batch_vars, **model_vars})

                    # we assume that loss is always a key in the dict returned by the criterion
                    loss = error_vars['loss']

                    self.optimizer.zero_grad()
                    loss.backward()
                    if kwargs['clip'] is not None:
                        self.clip_gradient(model, kwargs['clip'])
                    self.optimizer.step()

                    # record loss
                    error_vars['loss'] = error_vars['loss'].detach()
                    self.loss_sum += error_vars['loss'].detach().item()
                    logger.log(epoch, **error_vars)

                    # error_vars = self.iterate_model(model, criterion, batch_vars, logger, epoch, **kwargs)
                    self.print_minibatch_info(minibatch_number, num_minibatches, error_vars)
                    minibatch_number += 1
            mean_loss = self.loss_sum / minibatch_number
            # visual report for sanity
            print('\rEpoch:' + str(epoch) + ' > < ' + str(mean_loss) + ' '*100)
            model.save_model(training_dir, 'model_' + str(epoch))

        training_time = time.time() - t
        print('trained in ' + str(training_time) + ' seconds')
        model.eval()
        model.start_epoch = model.start_epoch + kwargs['epochs']
        return training_time

    def print_minibatch_info(self, minibatch_number, num_minibatches, error_vars):
        print(
            '\r    Minibatch:' + str(minibatch_number) + '/' + str(num_minibatches) + ' > < ' +
            self.print_dict(error_vars) + ' ' * 100, end=''
        )

    def iterate_model(self, model, criterion, batch_vars, logger, epoch, **kwargs):
        model_vars = model.forward(**batch_vars)
        error_vars = criterion(**{**batch_vars, **model_vars})

        # we assume that loss is always a key in the dict returned by the criterion
        loss = error_vars['loss']

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if kwargs['clip'] is not None:
            self.clip_gradient(model, kwargs['clip'])
        self.optimizer.step()

        # record loss
        error_vars['loss'] = error_vars['loss'].detach()
        self.loss_sum += error_vars['loss'].detach().item()
        logger.log(epoch, **error_vars)

        return error_vars

    @staticmethod
    def test(model, test_set, criterion, logger, **kwargs):
        model.eval()
        batch_size = utils.get_arg_default('batch_size', 100, **kwargs)
        crop = utils.get_arg_default('crop', 32, **kwargs)
        test_set.set_crop(crop)
        batch_vars = -1
        print()
        test_set.prepare_epoch()
        while batch_vars is not None:
            batch_vars = test_set.get_next_minibatch(batch_size)
            if batch_vars is not None:
                model_vars = model.forward(**{**batch_vars, **kwargs})
                error_vars = criterion(**{**batch_vars, **model_vars, **kwargs})
                # record loss
                error_vars['loss'] = error_vars['loss'].detach()
                logger.log(0, **error_vars)
