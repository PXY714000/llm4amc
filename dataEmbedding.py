import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class positionEmbedding(nn.Module):
    def __init__ (self,d_model,maxLength = 5000):
        super(positionEmbedding, self).__init__()
        pe = torch.zeros(maxLength,d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, maxLength).float().unsqueeze(1)  # 5000,1

        div_term = (torch.arange(0, d_model, 2).float()  # 256
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1,5000,512
        self.register_buffer('pe', pe)

    def forward(self,x):
        return self.pe[:,:x.size(1)]

class tokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(tokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        return x

class dataEmbedding(nn.Module):
    def __init__(self,c_in,d_model, dropout = 0.1):
        super(dataEmbedding,self).__init__()


        self.valueEmbedding = tokenEmbedding(c_in = c_in,d_model = d_model)
        self.positionEmbedding = positionEmbedding(d_model = d_model)

        self.dropout = nn.Dropout(p = dropout)


    def forward(self,x,xMark):
        result = self.valueEmbedding(x) + self.positionEmbedding(x)
        result = self.dropout(result)
        return result        