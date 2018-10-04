'''
name: LSTM, Long-Short Term Memory

Abstract: 

Author: Osvaldo Armas oosv@amazon.com

'''
from models.GenericModel import Model
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch
import torch.optim as optim

class LSTM(nn.Module, Model):
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
        '''
        Currently activation functions are not being used, this may 
        change depending on performance, but choise of function is
        very important in this particular problem, an alternate
        solution is to create an embedding for the cepstral coefficients
        '''
        #Linear
        self.linear1 = nn.Linear(self.feature_dim, self.hidden_dim)
        
        #Rectifying Linear Unit
        self.relu = nn.ReLU()
        
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
        out = self.relu(out)
        
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


    def train(self, training_set, language_idx, epoch=1, batch_size = 100, update=False):
        '''
        :param training_set: x_ins to train model over
        :param language_idx: language index corresponding to language model
        model assumes each utterance is independent of any other utterance,
        where utterance is a 10ms frame
        '''
        # Assumed that data input is in array format and
        # converted to Tensor. If already tensor nothing changes
        training_set = torch.tensor(training_set)
        language_idx = torch.tensor(language_idx, dtype=torch.int64)
        
        print(language_idx.size())
        # Data that's longer than the batch size could be padded,
        # instead we just disregard it
        # I will probably change that in the future
        extra_data = training_set.size(0) % batch_size
        if extra_data:
            training_set = training_set[:len(training_set) - extra_data]
            language_idx = language_idx[:len(language_idx) - extra_data]

        # reshape input array into batches
        # n_batches x batch_size x seq_length x n_features 
        training_set = torch.reshape(
            training_set,
            [int(len(training_set) / batch_size),
                batch_size, 
                training_set.size(1), 
                training_set.size(2)])
        
        # reshape input labels into batches
        # n_batches x batch_size x seq_length x 1 (language_id)
        # language_idx = torch.reshape(
        #     language_idx,
        #     [int(len(language_idx) / batch_size),
        #         batch_size, 
        #         language_idx.size(1), 
        #         language_idx.size(2)])
        
        # initialize loss function and optimizer
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.ASGD(self.parameters(), lr=0.5)
        
        # 
        loss_over_time = []
        epoch_loss = {}
        for e in range(epoch):
            for x_in, language in zip(training_set, language_idx):
                # zero_grad prevents training a new batch
                # on the last batch's gradient
                self.zero_grad()
                
                # this calls the forward function, through PyTorch
                # output in shape batch_size x 1 x n_languages
                scores = self(x_in)
                #print(scores.size())
                # calculate backward loss, get perform gradient descent (ASGD)
                loss = self.loss_function(scores,language.view(-1))
                loss.backward()
                self.optimizer.step()

                # for visualizing loss over time                
                loss_over_time.append(loss)
                if update:
                    print(f'loss: {loss.data}, epoch: {e} iter: {len(loss_over_time)}')
            epoch_loss[e] = np.average(loss.data)   
        
        return loss_over_time, epoch_loss
    
    def predict(self, X):
        '''
        :features: vector representing features of a single instance
        '''
        #May have to convert X to tensors
        X = torch.tensor(X)
        return self(X)

    def predict_all(self, x_list):
        '''
        :param x_list: list of x_ins x to predict
        '''
        pred = []
        for x in x_list:
            pred.append(self.predict(x))
        return pred

    def adjust_data_to_model(self, data):
        '''
        :param data: data to adjust to model expecting len(data)x3x75x13
        :return: tuple of data with adjusted dimensions and it's new shape
        '''
        return data

    