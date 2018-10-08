'''
name: LSTM, Long-Short Term Memory

Abstract: 

Author: Osvaldo Armas oosv@amazon.com

'''
import numpy as np
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


class LSTM(nn.Module):
    def __init__(self,
                n_features = 39,
                n_hidden = 512,
                languages = 2, 
                total_frames = 75,
                dropout=None,
                bidirectional=False,
                num_layers = 1,
                linear_layers = 1):
        '''
        :param n_features: number of features in a sample
        :param n_hidden: number of hidden dimensions to use
        :param languages: number of languages to score over
        :param total_frames: length of audio sample, sequence length
        :param dropout: dropout rate to use
        '''
        super(LSTM, self).__init__()

        # number of hidden dimensions, could be more complex
        # but we maintain the same number in all layers
        self.hidden_dim = n_hidden

        # number of features, aka input dimension
        self.feature_dim = n_features
        print(self.feature_dim)
        print(self.hidden_dim)
        # number of languages to score, aka output dimension
        # the + 1 is to have a score for a language that is not
        # from the training set OR it's pure silence
        self.n_languages = languages

        #lenght of audio frame, aka sequence length
        self.total_frames = total_frames

        # use not yet implemented as it requires multiple LSTM layers
        self.dropout = dropout

        # number of lstm layers
        self.num_layers = num_layers

        # number of direction of lstm (1 or 2)
        self.directions = 2 if bidirectional else 1

        # BiLSTM
        self.BiLSTM = bidirectional

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''
        Currently activation functions are not being used, this may 
        change depending on performance, but choise of function is
        very important in this particular problem, an alternate
        solution is to create an embedding for the cepstral coefficients
        '''
        #Linear 1
        self.linear1 = nn.Linear(self.feature_dim, self.hidden_dim)
        
        #Rectifying Linear Unit
        self.relu1 = nn.ReLU()
        
        # #Linear 2
        # self.linear2 = nn.Linear(self.hidden_dim, self.feature_dim)

        # #ReLU 2
        # self.relu2 = nn.ReLU()

        # #Linear 3
        # self.linear3 = nn.Linear(self.hidden_dim, self.feature_dim)

        # #ReLU 3
        # self.relu3 = nn.ReLU()

        # #Linear 4
        # self.linear4 = nn.Linear(self.hidden_dim, self.feature_dim)

        # #ReLU 4
        # self.relu4 = nn.ReLU()

        #Sigmoid Function
        self.sigmoid = nn.Sigmoid()

        #definition of lstm

        self.lstm = nn.LSTM(
            self.hidden_dim, 
            self.hidden_dim * self.directions, 
            batch_first = True,
            num_layers = self.num_layers,
            bidirectional = self.BiLSTM) 
        
        #converts LSTM output to languages
        self.language_scores = nn.Linear(self.hidden_dim, self.n_languages)
        
        # defining other supporting forms of data, loss and optimizer should beå
        self.hidden = self.init_hidden()

    def forward(self, x_in):
        '''
        :param x_in: sample audio
        :return: scores for audio sample
        '''
        
        # during the prediction step, the input dimensions will
        # be total_frames x n_features, the LSTM requires 3 dimensions
        # thus we reshape the sample to 1 x total_frames x n_features
        # implicitly (in case sequence is shorter than frame length)
        shape = x_in.size()
        if len(shape) < 3:
            x_in = x_in.reshape(1, shape[0], shape[1])
            shape = x_in.size()

        #Make sure previous state does not affect next state
        self.hidden = self.init_hidden()
        
        #input dimension listed before the function is executed
        #batch_length x total_frames x n_features
        out = self.linear1(x_in)
        
        #relu layer, does not affect shape
        out = self.relu1(out)

        # same as before for three stacked layers
        # out = self.linear2(x_in)
        # out = self.relu(out)
        # out = self.linear3(x_in)
        # out = self.relu(out)
        # out = self.linear4(x_in)


        #batch_length x total_frames x n_hidden
    
        out, self.hidden = self.lstm(out.view([-1, out.size(1), out.size(2)] , self.hidden))
        
        #sigmoid activation
        #out = self.sigmoid(out)
        
        #batch_length x 1 x n_hidden
        languages = self.language_scores(out[:,-1])

        #batch_length x 1 x n_languages
        return f.log_softmax(languages, dim=1)


    def init_hidden(self):
        '''
        Initialize hidden state
        '''
        return (torch.zeros(self.num_layers * self.directions, 1, self.hidden_dim),
                torch.zeros(self.num_layers * self.directions, 1, self.hidden_dim))        
    
    def predict(self, x):
        '''
        :features: vector representing features of a single instance
        '''
        super(LSTM, self).eval()
        #May have to convert X to tensors
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            return np.argmax(self.forward(x).numpy())

    def predict_all(self, x_list):
        '''
        :param x_list: list of x_ins x to predict
        '''
        pred = []
        for x in x_list:
            pred.append(self.predict(x))
        return pred
