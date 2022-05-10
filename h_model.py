import pandas as pd
import torch
from torch.nn.modules.activation import Sigmoid
import torch.nn.functional as F

from typing import Optional
from torch.autograd import Variable

class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims_lst):
        '''
                input_dim (int)
                output_dim (int)
                hidden_dims_lst (list, each element is a integer, indicating the hidden size)

        '''
        super(MLP, self).__init__()
        layer_size = len(hidden_dims_lst) + 1
        dims = [input_dim] + hidden_dims_lst + [output_dim]
        self.dims = dims
        self.predictor = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
        self.norm = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=dims[i+1], affine=False) for i in range(layer_size)])
        # self.norm = self.norm.double()

    def forward(self, v):
        batch_size = v.shape[0]
        # predict
        v = v.float()
        for i, l in enumerate(self.predictor):
            # v = F.relu(self.norm[i](l(v)))
            v = F.relu(l(v))
            # v = torch.nn.MaxPool1d(2, padding=1)(v.reshape(batch_size, int(self.dims[i+1]/2), 2)).flatten(1)
        return v


class CNN(torch.nn.Sequential):
    def __init__(self, num_feat=3550, num_filters=[128, 64, 32], kernels=[7, 7, 7], hidden_dim=32, normalization='batch', affine=True):
        super(CNN, self).__init__()
        in_ch = [2] + num_filters
        layer_size = len(num_filters)
        # self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=in_ch[i+1]) for i in range(layer_size)])
        self.conv = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=in_ch[i],
                                                            out_channels=in_ch[i+1],
                                                            kernel_size=kernels[i]) for i in range(layer_size)])
        self.conv = self.conv.double()
        self.layer_size = len(num_filters)
        self.normalization = normalization
        
        if normalization == 'batch':
            self.norm = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=in_ch[i+1], affine=affine) for i in range(self.layer_size)])
            self.norm = self.norm.double()
        elif normalization == 'instance':
            self.norm = torch.nn.ModuleList([torch.nn.InstanceNorm1d(num_features=in_ch[i+1], affine=affine) for i in range(self.layer_size)])
            self.norm = self.norm.double()
        
        n_size_p = self._get_conv_output((2, num_feat))
        self.fc1 = torch.nn.Linear(n_size_p, hidden_dim)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        if self.normalization is not None:
            for i in range(self.layer_size):
                x = F.relu(self.norm[i](self.conv[i](x.double())))
        else:
            for l in self.conv:
                x = F.relu(l(x.double()))
        # batch_size = v.size(0)
        # v = v.view(v.size(0), v.size(2), -1)
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v


class CNN_RNN(torch.nn.Sequential):
    def __init__(self, num_feat=3550, hidden_dim=32, num_filters=[128, 64, 32], num_kernels=[3, 3, 3], normalization='batch', affine=True,
                 Use_GRU_LSTM='GRU', RNN_layers=2, RNN_hidden_dim=64, bidirectional=True, device='cpu'):
        super(CNN_RNN, self).__init__()
        self.Use_GRU_LSTM = Use_GRU_LSTM
        self.RNN_layers = RNN_layers
        self.RNN_hidden_dim = RNN_hidden_dim
        self.bidirectional = bidirectional
        self.device = device
        in_ch = [2] + num_filters
        self.in_ch = in_ch
        kernels = num_kernels   
        self.layer_size = len(num_filters)
        self.normalization = normalization
        self.conv = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=in_ch[i],
                                                            out_channels=in_ch[i+1],
                                                            kernel_size=kernels[i]) for i in range(self.layer_size)])
        self.conv = self.conv.double()

        if normalization == 'batch':
            self.norm = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=in_ch[i+1], affine=affine) for i in range(self.layer_size)])
            self.norm = self.norm.double()
        elif normalization == 'instance':
            self.norm = torch.nn.ModuleList([torch.nn.InstanceNorm1d(num_features=in_ch[i+1], affine=affine) for i in range(self.layer_size)])
            self.norm = self.norm.double()
        
        n_size_p = self._get_conv_output((2, num_feat))

        if Use_GRU_LSTM == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=in_ch[-1],
                                        hidden_size=RNN_hidden_dim,
                                        num_layers=RNN_layers,
                                        batch_first=True,
                                        bidirectional=bidirectional)

        elif Use_GRU_LSTM == 'GRU':
            self.rnn = torch.nn.GRU(input_size=in_ch[-1],
                                    hidden_size=RNN_hidden_dim,
                                    num_layers=RNN_layers,
                                    batch_first=True,
                                    bidirectional=bidirectional)
        else:
            raise AttributeError('Please use LSTM or GRU.')
        direction = 2 if bidirectional else 1
        self.rnn = self.rnn.double()
        self.fc1 = torch.nn.Linear(
            RNN_hidden_dim * direction * n_size_p, hidden_dim)
        self.linear = torch.nn.Linear(in_features=self.RNN_hidden_dim, out_features=1)
        
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, self.in_ch[-1], -1).size(2)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        return x

    def forward(self, v):
        if self.normalization is not None:
            for i in range(self.layer_size):
                v = F.relu(self.norm[i](self.conv[i](v.double())))
        else:
            for l in self.conv:
                v = F.relu(l(v.double()))
        batch_size = v.size(0)
        v = v.view(v.size(0), v.size(2), -1)
        
        if self.Use_GRU_LSTM == 'LSTM':
            direction = 2 if self.bidirectional else 1
            h0 = torch.randn(self.RNN_layers * direction,
                             batch_size, self.RNN_hidden_dim).to(self.device)
            c0 = torch.randn(self.RNN_layers * direction,
                             batch_size, self.RNN_hidden_dim).to(self.device)
            v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
        else:
            # GRU
            direction = 2 if self.bidirectional else 1
            h0 = torch.randn(self.RNN_layers * direction,
                             batch_size, self.RNN_hidden_dim).to(self.device)
            v, hn = self.rnn(v.double(), h0.double())

        # v = torch.flatten(v, 1)
        # v = self.fc1(v.float())
        v = self.linear(v[0]).flatten()
        return v


if __name__ == "__main__":
    pass
