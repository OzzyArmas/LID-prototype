#!/usr/bin/env python

# Model importing
from models import lstm

# Important external libraries
import numpy as np
import sagemaker_containers
# built-in libraries
 
import argparse
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
import torch.nn.functional as F
import torch.utils.data

# important SageMaker predefined paths
prefix = '/opt/ml/'
input_path = prefix + 'input/data/'
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
train_channel = 'training'
training_path = os.path.join(input_path, train_channel)
eval_channel = 'validation'
eval_path = os.path.join(input_path, eval_channel)

logger = logging.getLogger('instance')
lvl = logging.WARNING
lvl_e = logging.ERROR

def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    
    logger.info("Get train data loader")
    
    # Pre shuffled data, x and y indeces matching
    train_data_x = np.load(os.path.join(training_path, 'train_x.npy' ))
    train_data_y = np.load(os.path.join(training_path, 'train_y.npy' )) 
    train_data_x = torch.tensor(train_data_x, dtype=torch.float32)
    train_data_y = torch.tensor(train_data_y, dtype=torch.int64)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_data_x = torch.utils.data.DataLoader(train_data_x, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler,
                                       **kwargs)

    train_data_y = torch.utils.data.DataLoader(train_data_y, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler,
                                       **kwargs)

    return train_data_x, train_data_y

def _get_test_data_loader(batch_size, training_dir, **kwargs):
    '''
    '''
    logger.info("Get train data loader")
    
    # Pre shuffled data, x and y indeces matching
    test_data_x = np.load(os.path.join(eval_path, 'test_x.npy' ))
    test_data_y = np.load(os.path.join(eval_path, 'test_y.npy' )) 
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y, dtype=torch.int64)

    test_data_x = torch.utils.data.DataLoader(test_data_x, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       **kwargs)

    test_data_y = torch.utils.data.DataLoader(test_data_y, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       **kwargs)
         
    return test_data_x, test_data_y


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
                                        is_distributed)
    test_x, test_y = _get_test_data_loader(args.test_batch_size, 
                                        args.data_dir)
    
    model = lstm.MixedLSTM(
                    n_features = args.n_features, 
                    n_hidden = args.n_hidden, 
                    languages = args.languages,
                    total_frames = args.frames, 
                    dropout = args.dropout, 
                    bidirectional = args.bidirectional,
                    num_layers = args.num_layers,
                    linear_layers = args.linear_layers
                    ).to(device)
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr)
    
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
                    epoch, batch_idx * len(feature_seq), len(train_x.sampler),
                    100. * batch_idx / len(train_x.sampler), loss.item()))
        
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

    test_loss /= len(test_x)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_x),
        100. * correct / len(test_x)))    
    
def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        
        # Data and Training information
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
        
        # Model Specific Details
        parser.add_argument('--n_features', type=int, default=39,
                            help='number of features')
        parser.add_argument('--n_hidden', type=int, default=512,
                            help='number of hidden layers')
        parser.add_argument('--languages', type=int, default=2,
                            help='number of languages to learn')
        parser.add_argument('--frames', type=int, default=150,
                            help='total frames per sample/utterance')
        parser.add_argument('--dropout', type=float, default=None,
                            help='desired dropout')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='total number of lstm layers')
        parser.add_argument('--bidirectional', type=bool, default=False,
                            help='use bidirectional lstm')
        parser.add_argument('--linear_layers', type=int, default=1,
                            help='number of linear layers')

        # Container environment
        env = sagemaker_containers.training_env()
        parser.add_argument('--hosts', type=list, default=env.hosts)
        parser.add_argument('--current-host', type=str, default=env.current_host)
        parser.add_argument('--model-dir', type=str, default=env.model_dir)
        parser.add_argument('--data-dir', type=str,
                            default=env.channel_input_dirs['training'])
        parser.add_argument('--num-gpus', type=int, default=env.num_gpus)
        
        train(parser.parse_args())

    except Exception as e:
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as err:
            err.write('Exception during training: ' + str(e) + '\n' + trc)
            logger.error('Exception during training: ' + str(e) + '\n' + trc)
        print('Exception during training: ' + str(e) + '\n' + trc)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(1)