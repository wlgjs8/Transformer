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
        output, _ = self.MultiHeadAttention(input, input, input, mask=mask)
        output = output.view(1, 2050)
        output = self.PositionwiseFeedForward(output)
        # output = output + output

        return output


class token_transformation(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        # self.bbox_token = nn.Parameters(torch.zeros(1, 1, d_model))
        # self.segm_token = nn.Parameters(torch.zeros(1, 1, d_model))
        self.bbox_token = torch.zeros(1)
        self.segm_token = torch.zeros(1)

    def forward(self, enc_src):

        self.bbox_token = self.bbox_token.cuda()
        self.segm_token = self.segm_token.cuda()

        enc_src = torch.cat([self.bbox_token, enc_src, self.segm_token], dim = 0)

        return enc_src

# class DecoderLayer(nn.Module):
#     def __init__(feat, bbox_token, segm_token):



class Transformer(nn.Module):
    def __init__(self, n_head, d_key, d_value, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, 100)
        self.softmax = nn.LogSoftmax(dim=1)

        self.token_transformation = token_transformation(d_model)
        self.encoder = EncoderLayer(d_model, d_inner, n_head, d_key, d_value, dropout=dropout)
        # self.decoder = DecoderLayer()

    def forward(self, enc_src):
        output = self.token_transformation(enc_src)
        output = self.encoder(output)
        output = self.fc(output)
        output = self.softmax(output)

        return output