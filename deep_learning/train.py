#!/usr/bin/env python

# Model importing
from models import lstm

# Important external libraries
import pandas as pd
import numpy as np

# built-in libraries
import os
import traceback
import sys
import pickle
import json
import time
import ast
import logging

# Pytorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

# important SageMaker predefined paths
prefix = '/opt/ml/'
input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

'''
setting training path, there's other channels for
evaluation and testing. Generally these only useful when 
the training uses PIPE mode instead of FILE mode; lstm 
model is currently set up to use FILE as there isn't
incredibily large amounts of data yet. PIPE should most definitely 
be used when training data exceeds 10 GB
'''
channel = 'training'
training_path = os.path.join(input_path, channel)
train_data = os.path.join(training_path, 'train')
test_data = os.path.join(training_path, 'test')

logger = logging.getLogger('instance')
lvl = logging.WARNING
lvl_e = logging.ERROR

def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    
    logger.info("Get train data loader")
    
    # Pre shuffled data, x and y indeces matching
    train_data_x = np.load(os.path.join(training_path, 'train_x.npy' ))
    train_data_y = np.load(os.path.join(training_path, 'train_y.npy' )) 
    
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_data_x = torch.utils.data.DataLoader(train_data_x, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler,
                                       **kwargs)

    train_data_y torch.utils.data.DataLoader(train_data_y, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler,
                                       **kwargs)

    return train_data_x, train_data_y

def _get_test_data_loader(batch, dir, distributed, **kwargs):
    '''
    '''
    logger.info("Get train data loader")
    
    # Pre shuffled data, x and y indeces matching
    test_data_x = np.load(os.path.join(training_path, 'test_x.npy' ))
    test_data_y = np.load(os.path.join(training_path, 'test_y.npy' )) 
    
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    test_data_x = torch.utils.data.DataLoader(test_data_x, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler,
                                       **kwargs)

    test_data_y torch.utils.data.DataLoader(test_data_y, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler,
                                       **kwargs)
         
    return train_data_x, train_data_y


def train(args, model_params):
    
    # Get parameters provided by SageMaker
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, 
                                rank=host_rank, 
                                world_size=world_size)
        logger.info(
            'Init distributed env: \'{}\' backend on {} nodes. '.format(args.backend, 
                dist.get_world_size()) + \
            'Current host rank is {}. Number of gpus: {}'.format(
                dist.get_rank(), args.num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_x, train_y = _get_train_data_loader(args.batch_size, 
                                        args.data_dir, 
                                        is_distributed,
                                        **kwargs)
    test_x, test_y = _get_test_data_loader(args.test_batch_size, 
                                        args.data_dir,
                                        **kwargs)
    model = create_model(model_params, device)
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        for batch_idx, (feature_seq, language) in enumerate(zip(train_x, train_y),1):
            # indicate to use this data with GPU
            feature_seq = feature_seq.to(device)
            language = language.to(device)
            
            # zero_grad prevents training a new batch
            # on the last batch's gradient
            optimizer.zero_grad()

            # this calls the forward function, through PyTorch
            # output in shape batch_size x 1 x n_languages
            scores = model(feature_seq)

            # calculate backward loss, get perform gradient descent
            loss = loss_function(scores, language.view(-1))
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()

            # update logging information
            if batch_idx % args.log_interval == 0:
                logger.info('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        test(model, test_x, test_y, device)
        
    save_model(model, args.model_dir)


def test(model, test_x, test_y, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in zip(test_x, test_y):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    
    
def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

def create_model(model_params, device='cpu'):
    '''
    :param languages: languages to train lstm over
    :param model_params: parameters to use for training LSTM
        defaults:
            n_features = 39,
            n_hidden = 512,
            languages = 2, 
            snippet_length = 75,
            dropout=0.0,
            bi_directional=False,
            num_layers = 1,
            linear_layers = 1)
    :return: a LSTM model object containing a lstm per language
    '''    
    # the json file may convert the array given in the hyperparams json into a 
    # a string, so this is just to circumvent that problem
    if isinstance(clusters, str):
        clusters = ast.literal_eval(clusters)
    
    # apparently hyperparms loves to give strings, thus int conversions
    n_features      =   int(model_params.get('n_features', 39)),
    n_hidden        =   int(model_params.get('n_hidden', 512))
    languages       =   int(model_params.get('languages', 2)) 
    snippet_length  =   int(model_params.get('snippet_length', 75))
    dropout         =   int(model_params.get('dropout', 0.0))
    num_lstm_layers =   int(model_params.get('num_layers', 1))
    bidirectional   =   bool(model_params.get('bidirectional', False))

    # if there is a gpu, the LSTM will take care of checking for that during training
    return lstm.LSTM(
        n_features,
        n_hidden,
        languages,
        snippet_length,
        dropout,
        bidirectional,
        num_lstm_layers
    ).to(device)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)

if __name__ == '__main__':
    try:

        with open(param_path, 'r') as hyper:
            model_params = json.load(hyper)
        
        parser = argparse.ArgumentParser()

        # Data and model checkpoints directories
        parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=2, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.001)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--backend', type=str, default=None,
                            help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

        # Container environment
        env = sagemaker_containers.training_env()
        parser.add_argument('--hosts', type=list, default=env.hosts)
        parser.add_argument('--current-host', type=str, default=env.current_host)
        parser.add_argument('--model-dir', type=str, default=env.model_dir)
        parser.add_argument('--data-dir', type=str,
                            default=env.channel_input_dirs['training'])
        parser.add_argument('--num-gpus', type=int, default=env.num_gpus)
        
        train(parser.parse_args(), model_params)

    except Exception as e:
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as err:
            err.write('Exception during training: ' + str(e) + '\n' + trc)
            logger.error('Exception during training: ' + str(e) + '\n' + trc)
        print('Exception during training: ' + str(e) + '\n' + trc)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(1)