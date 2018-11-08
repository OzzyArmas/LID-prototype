'''
name: LSTM, Long-Short Term Memory

Abstract: 

Mixed LSTM model allowing for linear layers to be added before LSTM

Author: Osvaldo Armas oosv@amazon.com

'''
import numpy as np
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from collections import OrderedDict

class MixedLSTM(nn.Module):
    def __init__(self,
                n_features = 39,
                n_hidden = 512,
                languages = 2, 
                total_frames = 75,
                dropout=0,
                bidirectional=False,
                lstm_layers = 1):
        '''
        :param n_features: number of features in a sample
        :param n_hidden: number of hidden dimensions to use
        :param languages: number of languages to score over
        :param total_frames: length of audio sample, sequence length
        :param dropout: dropout rate to use
        :param bidirectional: whether to use a BiLSTM or not
        :param lstm_layers: number of lstm layers
        '''
        super(MixedLSTM, self).__init__()

        '''
        conv_nets about to be added
        '''
        # number of hidden dimensions, could be more complex
        # but we maintain the same number in all layers
        self.hidden_dim = n_hidden

        # number of features, aka input dimension
        self.feature_dim = n_features
        
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

        # main Linear Layer
        self.linear_main = nn.Linear(self.feature_dim, self.hidden_dim)
        
        # Sigmoid Function
        self.sigmoid_main = nn.Sigmoid()
        
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

    def forward(self, x_in):
        '''
        :param x_in: sample audio
        :return: scores for audio sample
        '''
        # make sure previous state (prediction) does not affect next state
        self.hidden = self.init_hidden()
        
        # input dimension listed before the function is executed
        # batch_length x total_frames x n_features
        out = self.linear_main(x_in)

        # relu layer, does not affect shape
        # batch_length  x total_frames x n_hidden
        out = self.sigmoid_main(out)

        # batch_length x total_frames x n_hidden
        out, self.hidden = self.lstm(out.view([-1,
                                        out.size(1),
                                        out.size(2)],
                                        self.hidden))
        
        # batch_length x 1 x n_hidden, only use scores from last state
        languages = self.language_scores(out[:,-1])

        # batch_length x 1 x n_languages
        return f.log_softmax(languages, dim=1)


    def init_hidden(self):
        '''
        Initialize hidden state of lstm
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
        super(MixedLSTM, self).eval()
        # May have to convert X to tensors
                
        # during the prediction step, the input dimensions will
        # be total_frames x n_features, the LSTM requires 3 dimensions
        # thus we reshape the sample to 1 x total_frames x n_features
        # implicitly in case sequence is shorter than frame length

        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            shape = x.size()
            if len(shape) < 3:
                x = x.reshape(1, shape[0], shape[1])
            
            return np.argmax(self.forward(x).numpy())

    def predict_all(self, x_list):
        '''
        :param x_list: list of x_ins x to predict
        '''
        pred = []
        for x in x_list:
            pred.append(self.predict(x))
        return pred

