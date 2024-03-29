#!/usr/bin/env python

# package imports
from models import convlstm
from models import lstm
from utils import metrics

# This file only works within a sagemaker container
import sagemaker_containers

# Important external libraries
import numpy as np

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
from shutil import copyfile
from collections import defaultdict

# Pytorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist

# important SageMaker predefined paths
prefix = '/opt/ml/'
input_path = prefix + 'input/data/'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

# Training Data
train_channel = 'training'
training_path = os.path.join(input_path, train_channel)

# Validation Data
eval_channel = 'validation'
eval_path = os.path.join(input_path, eval_channel)

# Logging used... I'm still having some issues with this but
# it's low priority as important metadata is being saved
logger = logging.getLogger()

# Start timing how long will the job run for
start = time.time()

# Some CONSTANTS
TRUTH_VALUES = {'yes', 'true', 't', 'y', '1'}
FALSE_VALUES = {'no', 'false', 'f', 'n', '0'}

# file name:
TRAIN_X      = 'train_x.npy'
TRAIN_Y      = 'train_y.npy'
TEST_X       = 'test_x.npy'
TEST_Y       = 'test_y.npy'
FULL_TRAIN_X = 'full_train_x.npy'
FULL_TRAIN_Y = 'full_train_y.npy'
FULL_TEST_X  = 'full_test_x.npy'
FULL_TEST_Y  = 'full_test_y.npy'


CHANNELS =  3
CONVLSTM = 'ConvLSTM'
MIXEDLSTM = 'MixedLSTM'

# The values below are hardcoded to represent current voxforge data sets being
# used for 6 and 5 languages

# When using 6 languages
# LANGUAGE_DISTRIBUTION = torch.tensor([ # Fractions of language representation
#                                         0.21185058424022032,  # English
#                                         0.21268393985578937,  # Spanish
#                                         0.21167232635453712,  # French
#                                         0.12096580122463167,  # Italian
#                                         0.12117971068745154,  # Russian
#                                         0.12164763763736998], # German
#                                         dtype=torch.float32)

# When using 5 languages
LANGUAGE_DISTRIBUTION = torch.tensor([
                                        0.3941189279890582,  # English
                                        0.23150166450243723,  # Spanish
                                        0.15468183642212452,  # French
                                        0.08698454499837227,  # Italian
                                        0.13271302608800778], # German
                                        dtype=torch.float32)

def _average_gradients(model):
    '''
    :param model: nn.Module representing model after a batch.
        Gradiant Averaging gets the average gradient accross workers
        and uses it to calculate gradiant descent
    '''
    # Gradient averaging.
    size = float(dist.get_world_size())
    
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def _get_train_data_loader(batch_size, file_x, file_y, model, is_distributed, **kwargs):
    '''
    :param batch_size: batch size to separate data into
    :param training_dir: directory to get data from,
        may require different implementation if using SageMaker PIPE mode
    :param is_distributed: whether or not distributed learning is being used
        (a.k.a accross more than one instance)
    :param **kwargs: any DataLoader specific kwargs
    '''
    logger.warning("Get train data loader")
    
    # Pre shuffled data, x and y indeces matching
    train_data_x = np.load(os.path.join(training_path, file_x))
    train_data_y = np.load(os.path.join(training_path, file_y)) 
    train_data_x = torch.tensor(train_data_x, dtype=torch.float32)
    train_data_y = torch.tensor(train_data_y, dtype=torch.int64)
    
    shape_x = train_data_x.size()  
    if model == CONVLSTM:
        if len(shape_x) > 3:
            # shape is Samples x channels x frames x coef
            # and converted to Samples x channels x coef x frames
            train_data_x = train_data_x.reshape(shape_x[0],
                                                shape_x[1],
                                                shape_x[3],
                                                shape_x[2])
        else:
            # shape is Number Samples x frames x coefficients
            # and is converted to Samples x channels x coeff x frames
            train_data_x = train_data_x.reshape(shape_x[0],
                                                CHANNELS,
                                                shape_x[2] // CHANNELS,
                                                shape_x[1])

    elif len(shape_x) > 3:
        # shape is Samples x channels x frames x coef
        # and is converted to Samples x Frames x Channels * Coeff
        train_data_x = train_data_x.reshape(shape_x[0],
                            shape_x[2],
                            shape_x[3] * shape_x[1])

    if is_distributed:
        train_sampler_x = \
            torch.utils.data.distributed.DistributedSampler(train_data_x)
        train_sampler_y = \
            torch.utils.data.distributed.DistributedSampler(train_data_y)

    else:
        train_sampler_x = None
        train_sampler_y = None


    train_data_x = torch.utils.data.DataLoader(train_data_x, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler_x,
                                       **kwargs)

    train_data_y = torch.utils.data.DataLoader(train_data_y, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       sampler=train_sampler_y,
                                       **kwargs)

    return train_data_x, train_data_y


def _get_test_data_loader(batch_size, file_x, file_y, model, **kwargs):
    '''
    :param batch_size: batch size to separate data into
    :param training_dir: directory to get data from,
        may require different implementation if using SageMaker PIPE mode
    :param **kwargs: any DataLoader specific kwargs
    '''
    logger.warning("Get test data loader")
    
    # Pre shuffled data, x and y indeces matching
    test_data_x = np.load(os.path.join(eval_path, file_x))
    test_data_y = np.load(os.path.join(eval_path, file_y)) 
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y, dtype=torch.int64)    

    shape_x = test_data_x.size()  
    if model == CONVLSTM:
        if len(shape_x) > 3:
            # shape is Samples x channels x frames x coef
            # and converted to Samples x channels x coef x frames
            test_data_x = test_data_x.reshape(shape_x[0],
                                              shape_x[1],
                                              shape_x[3],
                                              shape_x[2])
        else:
            # shape is Number Samples x frames x coefficients
            # and is converted to Samples x channels x coeff x frames 
            test_data_x = test_data_x.reshape(shape_x[0],
                                              CHANNELS,
                                              shape_x[2] // CHANNELS,
                                              shape_x[1])
    elif len(shape_x) > 3:
        # shape is Samples x channels x frames x coef
        # and is converted to Samples x Frames x Channels * Coeff
        test_data_x = test_data_x.reshape(shape_x[0],
                                          shape_x[2],
                                          shape_x[3] * shape_x[1])

    test_data_x = torch.utils.data.DataLoader(test_data_x, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       **kwargs)

    test_data_y = torch.utils.data.DataLoader(test_data_y, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       **kwargs)
         
    return test_data_x, test_data_y


def _get_model(args, device):
    '''
    :param args: arguments to pass in to model
    :returns: a model using correct arguments
    '''
    if args.model == CONVLSTM:
        return convlstm.ConvLSTM(
                        n_features = args.n_features, 
                        n_hidden = args.n_hidden, 
                        languages = args.languages,
                        total_frames = args.frames, 
                        dropout = args.dropout, 
                        bidirectional = args.bidirectional,
                        lstm_layers = args.lstm_layers,
                        kernel = args.kernel,
                        out_channels = args.output_channels
                        ).to(device)

    # defaults to MixedLSTM
    return lstm.MixedLSTM(
                    n_features = args.n_features, 
                    n_hidden = args.n_hidden, 
                    languages = args.languages,
                    total_frames = args.frames, 
                    dropout = args.dropout, 
                    bidirectional = args.bidirectional,
                    lstm_layers = args.lstm_layers
                    ).to(device)


def train(args):
    '''
    :param args: argument object containing values of commandline args
    '''
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
        logger.warning(
            'Init distributed env: \'{}\' backend on {} nodes. '.format(args.backend, 
                dist.get_world_size()) + \
            'Current host rank is {}. Number of gpus: {}'.format(
                dist.get_rank(), args.num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_x, train_y = _get_train_data_loader(args.batch_size,
                                        TRAIN_X,
                                        TRAIN_Y,
                                        args.model, 
                                        is_distributed)
    
    test_x, test_y = _get_test_data_loader(args.test_batch_size,
                                        TEST_X,
                                        TEST_Y,
                                        args.model)
    
    # get the correct model to train
    model = _get_model(args, device)
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    # This specific weight value should only be used when training on the 
    # 6 Language Dataset made by oosv@amazon.com
    loss_function = nn.NLLLoss(weight = LANGUAGE_DISTRIBUTION.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.to(device)
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

            # calculate loss and perform gradient descent using ADAM
            loss = loss_function(scores, language.view(-1))
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()

            # update logging information
            if batch_idx % args.log_interval == 0:
                logger.debug(
                    'Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(feature_seq), len(train_x.sampler),
                    100. * batch_idx / len(train_x.sampler), loss.item()))

        acc = test(model, args.languages, test_x, test_y, device, epoch, best_acc)
        # save model with best accuracy
        if best_acc < acc:
            save_model(model, args.model_dir)
            best_acc = acc


    # the section below is reserved for training on full length (padded) utterances
    full_train_x, full_train_y = _get_train_data_loader(args.batch_size, 
                                        FULL_TRAIN_X,
                                        FULL_TRAIN_Y,
                                        args.model, 
                                        is_distributed)
    
    full_test_x, full_test_y = _get_test_data_loader(args.batch_size, 
                                        FULL_TEST_X,
                                        FULL_TEST_Y,
                                        args.model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr / 10)
    loss_function = nn.NLLLoss(weight = LANGUAGE_DISTRIBUTION.to(device))
    for epoch in range(args.epochs + 1, args.epochs + args.epochs // 2):
        model.to(device)
        model.train()
        
        for batch_idx, (feature_seq, language) in enumerate(zip(full_train_x, full_train_y),1):
            # indicate to use this data with GPU
            feature_seq = feature_seq.to(device)
            language = language.to(device)
            
            # zero_grad prevents training a new batch
            # on the last batch's gradient
            optimizer.zero_grad()

            # this calls the forward function, through PyTorch
            # output in shape batch_size x 1 x n_languages
            scores = model(feature_seq)

            # calculate loss and perform gradient descent using ADAM
            loss = loss_function(scores, language.view(-1))
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()

            # update logging information
            if batch_idx % args.log_interval == 0:
                logger.debug(
                    'Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(feature_seq), len(full_train_x.sampler),
                    100. * batch_idx / len(full_train_x.sampler), loss.item()))

        acc = test(model, args.languages, full_test_x, full_test_y, device, epoch, best_acc)
        # save model with best accuracy
        if best_acc < acc:
            save_model(model, args.model_dir)
            best_acc = acc
    


def test(model, languages, test_x, test_y, device, epoch, best_acc):
    '''
    :param model: nn.Model representing most recently trained version of model
    :param languages: total languages model is scoring over
    :param test_x: DataLoader for features
    :param test_y: DataLoader for labels
    :param device: torch.device to use, cpu/cuda
    :param epoch: current Epoch of training
    :param best_acc: best accuracy to use indicate which file holds the best
        models performance metadata
    :returns: accuracy of current test to be used to saved best model
    '''
    # model.eval() prevents model from accumulating gradients during testing
    model.eval()
    test_loss = 0
    correct = 0
    
    # Start dictionaries to keep track of
    # False Acceptance Rate and False Rejection Rate
    FAR = defaultdict(float)
    FRR = defaultdict(float)

    # Initialize a confusion matrix in the form of an ndarray
    conf_matrix = np.zeros((languages, languages))

    # Do not Calculate gradient when evaluating/testing data
    with torch.no_grad():
        for data, target in zip(test_x, test_y):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            
            # Calculate indeces representing wrong predictions
            target = target.view_as(pred)
            equality_vec = pred.eq(target)
            correct += equality_vec.sum().item()
            
            # Wrong values, get False Acceptances, False Rejections
            for idx in (equality_vec == 0).nonzero():
                FAR[pred[idx][0].item()] += 1
                FRR[target[idx][0].item()] += 1

            for idx,p in enumerate(pred):
                conf_matrix[target[idx][0].item()][pred[idx][0].item()] += 1

    # Convert the counted false acceptances and rejections
    # into rates, then calculate EER
    FAR, FRR, EER = metrics.get_error_rates(FAR, FRR, len(test_x.dataset))

    # get recall and precision using confusion matrix
    recall = metrics.get_recall(conf_matrix)
    precision = metrics.get_precision(conf_matrix)

    # Test metadata to save
    test_loss /= len(test_x.dataset)
    file_name = 'accuracy_{}.json'.format(epoch)
    end = time.time()
    metadata = {
        'test_loss'         : test_loss,
        'acc'               : correct/len(test_x.dataset),
        'time'              : end - start,
        'epoch'             : epoch,
        'FAR'               : FAR,
        'FRR'               : FRR,
        'EER'               : EER,
        'total_correct'     : correct,
        'total_test_set'    : len(test_x.dataset),
        'recall'            : recall,
        'precision'         : precision, 
        'f-score'           : metrics.get_fscore(precision, recall),
        'confusion'         : conf_matrix.tolist()
        }
    
    with open(os.path.join(model_path, file_name), 'w') as out:
        json.dump(metadata, out, indent="\t")    
    
    # save the best epoch metadata in a separate file and replace
    # existing file
    if best_acc < metadata['acc']:
        best_file = 'best_model.json'
        with open(os.path.join(model_path, best_file), 'w') as out:
            json.dump(metadata, out, indent="\t")

    return metadata['acc']


def save_model(model, model_dir):
    '''
    :param model: nn.Module to save
    :param model_dir: SageMaker specified directory to save output data into 
    '''
    logger.warning("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)


def bool_parse(str_input):
    '''
    :param str_input: string input to verify if it should
        evaluate to TRUE or FALSE
    :returns: whether or not the str_input correctly evaluates to
        TRUE or FALSE
    :raises ArgumentTypeError: in case input is not withing either set
        of possible values
    '''
    if str_input.lower() in TRUTH_VALUES:
        return True
    elif str_input.lower() in FALSE_VALUES:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tuple_parse(str_input):
    '''
    :param str_input: string input to verify if the value is an input
        tuple or not, if not an input, ast.literal_eval will take care
        of converting it
    :returns: whether or not the str_input correctly evaluates to
        TRUE or FALSE
    :raises ArgumentTypeError: In case a value other than a tuple or string
        was input, if the value is parse correctly, then ConvLSTM will raise
        an error if the numeric values are wrong
    '''
    if type(str_input) == tuple:
        return str_input
    elif type(str_input) == str:
        return ast.literal_eval(str_input)
    else:
        raise argparse.ArgumentTypeError('Tuple or String expected')


def get_parser():
    '''
    :returns an argument parser
    '''
    parser = argparse.ArgumentParser()
    
    # Data and Training information
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
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
    parser.add_argument('--model', type=str, default='MixedLSTM',
                        help='specify model as either MixedLSTM or ConvLSTM')
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
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='total number of lstm layers')
    parser.add_argument('--bidirectional', type=bool_parse, default=False,
                        help='use bidirectional lstm')
    parser.add_argument('--linear_layers', type=int, default=1,
                        help='number of linear layers')
    parser.add_argument('--kernel', type=tuple_parse, default=(1,7),
                        help='shape of kernel to apply on image')
    parser.add_argument('--output_channels', type=int, default=3,
                        help='number of output channels/filters to convolve over')

    # Container environment, basically this is taken car of by sage-maker
    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str,
                        default=env.channel_input_dirs['training'])
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    return parser


if __name__ == '__main__':
    try:
        parser = get_parser()
        train(parser.parse_args())
        param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
        copyfile(param_path, os.path.join(model_path, 'model_params.json'))

    except Exception as e:
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as err:
            err.write('Exception during training: ' + str(e) + '\n' + trc)
            logger.error('Exception during training: ' + str(e) + '\n' + trc)
        print('Exception during training: ' + str(e) + '\n' + trc)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(1)