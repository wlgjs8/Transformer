import math
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
import numpy as np
from model.sublayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    # def __init__(self, hidden_dim, num_head, inner_dim):
    def __init__(self, d_model, d_inner, n_head, d_key, d_value, dropout=0.1):
        super().__init__()

        # self.hidden_dim = hidden_dim
        # self.num_head = num_head
        # self.inner_dim = inner_dim

        # self.MultiHeadAttention = MultiHeadAttention(num_head, inner_dim, inner_dim, inner_dim)
        self.MultiHeadAttention = MultiHeadAttention(n_head, d_model, d_key, d_value, dropout=dropout)
        self.PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        

    def forward(self, input, mask=None):
        output = self.MultiHeadAttention(input, input, input, mask=mask)
        output_ = self.PositionwiseFeedForward(output)
        # output = output + output

        return output


class Transformer(nn.Module):
    def __init__(self, n_head, d_key, d_value, d_model, d_inner, dropout=0.1):
        super().__init__()
        # self.encoder = EncoderLayer(hidden_dim, num_head, inner_dim)
        self.encoder = EncoderLayer(d_model, d_inner, n_head, d_key, d_value, dropout=dropout)
        self.d_model = d_model

    def forward(self, enc_src):
        output = self.encoder(enc_src)

        return output