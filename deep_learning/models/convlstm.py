'''
Conv LSTM model allowing for linear layers to be added before LSTM

Author: Osvaldo Armas oosv@amazon.com

'''
import numpy as np
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from collections import OrderedDict

class ConvLSTM(nn.Module):
    def __init__(self,
                n_features      = 13,
                n_hidden        = 512,
                languages       = 2, 
                total_frames    = 150,
                dropout         = 0,
                bidirectional   = False,
                lstm_layers     = 1,
                kernel          = (1,7),
                out_channels    = 3):
        '''
        :param n_features: number of features in a sample
        :param n_hidden: number of hidden dimensions to use
        :param languages: number of languages to score over
        :param total_frames: length of audio sample, sequence length
        :param dropout: dropout rate to use
        :param bidirectional: whether to use a BiLSTM or not
        :param lstm_layers: number of lstm layers
        '''
        super(ConvLSTM, self).__init__()

        '''
        conv_nets about to be added
        '''
        # Input Channels is always going to be three
        self.IN_CHANNELS = 3
        
        # number of hidden dimensions, could be more complex
        # but we maintain the same number in all layers
        self.hidden_dim = n_hidden

        # number of features, aka input dimension
        self.feature_dim = n_features * out_channels
        
        # number of languages to score, aka output dimension
        self.n_languages = languages

        # lenght of audio frame, aka sequence length
        self.total_frames = total_frames

        # define drop out
        self.dropout = dropout

        # number of lstm layers
        self.lstm_layers = lstm_layers

        # BiLSTM
        self.BiLSTM = bidirectional

        # number of channels/filters to apply during convolution
        self.out_channels = out_channels
        
        # tuple representing the shape of the kernel to use for each channel
        self.kernel = kernel 
                
        # batch_size x channels x coefficients x frames
        self.conv2d = nn.Conv2d(
                            self.IN_CHANNELS,
                            out_channels,
                            self.kernel,
                            padding = (self.kernel[0]//2, self.kernel[1]//2)
                        )

        # main Linear Layer
        self.linear = nn.Linear(self.feature_dim , self.hidden_dim)
        
        # main Rectifying Linear Unit
        self.sigmoid = nn.Sigmoid()
        
        # definition of lstm
        self.lstm = nn.LSTM(
            input_size = self.hidden_dim, 
            hidden_size = self.hidden_dim // (1 + self.BiLSTM),
            batch_first = True,
            num_layers = self.lstm_layers,
            bidirectional = self.BiLSTM,
            dropout = self.dropout) 
        
        # converts LSTM output to languages
        self.language_scores = nn.Linear(self.hidden_dim, self.n_languages)
        
        # initialize hidden layer
        self.hidden = self.init_hidden()

    def forward(self, x):
        '''
        :param x: sample audio
        :return: scores for audio sample
        '''
        # make sure previous state (prediction) does not affect next state
        # may have te be changed to adjust for streaming data, as the forward
        # function will be called every step
        self.hidden = self.init_hidden()

        # batch_size x channels x n_coefficients x total_frames
        out = self.conv2d(x)
        
        # reshape into batch_size x total_frames x channel * n_coefficients
        out = out.reshape([out.size(0), out.size(3), out.size(1) * out.size(2)])

        # batch_length x total_frames x n_hidden
        out, self.hidden = self.lstm(out)
        
        # batch_length x 1 x n_hidden, only use scores from last state
        languages = self.language_scores(out[:,-1])

        # batch_length x 1 x n_languages
        return f.log_softmax(languages, dim=1)


    def init_hidden(self):
        '''
        Initialize hidden state of lstm (hidden and forget gates)
        dimensions are 2 x lstm_layers / directions x 1 x hidden_dim
        '''
        return (torch.zeros(
                    self.lstm_layers // (1 + self.BiLSTM),
                    1, self.hidden_dim),
                torch.zeros(
                    self.lstm_layers //  (1 + self.BiLSTM),
                    1, self.hidden_dim))        
    
    def predict(self, x):
        '''
        :param x: vector representing features of a single instance
        '''
        super(ConvLSTM, self).eval()
        # May have to convert X to tensors
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            
            # x must be shape:
            # batch_size x channels x n_coefficients x frames
            if not len(x.size()) == 4:
                raise Exception("Data is not in the correct dimensions, \
                correct dimensions should be \
                batch_size x channels x coefficients x frequencies")

            return np.argmax(self.forward(x).numpy())

    def predict_all(self, x_list):
        '''
        :param x_list: list of x_ins x to predict
        '''
        pred = []
        for x in x_list:
            pred.append(self.predict(x))
        return pred