import torch
from torch.autograd import Variable
import torch.nn as nn


class StackedRNN(nn.Module):
    """Stacked recurrent neural network
    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_classes : int
        Set number of classes.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    logsoftmax : bool, optional
        Defaults to True (i.e. apply log-softmax).
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[32, 32], bidirectional=True,
                 linear=[32]):

        super(StackedRNN, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.linear = linear

        self.num_directions_ = 2 if self.bidirectional else 1

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional,
                                         batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        # define post-linear activation
        self.tanh_ = nn.Tanh()
        self.sigmoid_ = nn.Sigmoid()

        # create final classification layer (with log-softmax activation)
        self.final_layer_ = nn.Linear(input_dim, self.n_classes)

    def get_loss(self):
        if self.logsoftmax:
            return nn.NLLLoss()
        else:
            return nn.CrossEntropyLoss()

    def forward(self, sequence):

        # check input feature dimension
        batch_size, n_samples, n_features = sequence.size()
        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence
        device = sequence.device

        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_, batch_size, hidden_dim,
                    device=device, requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (output[:, :, :hidden_dim] + \
                               output[:, :, hidden_dim:])

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):
            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = self.tanh_(output)

        # apply final classification layer
        output = self.final_layer_(output)
        output = self.sigmoid_(output)

        return output
