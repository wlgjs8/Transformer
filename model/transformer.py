import math
import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
import numpy as np
from model.sublayers import MultiHeadAttention, PositionwiseFeedForward
# from einops import rearrange, repeat
from einops import repeat

class EncoderLayer(nn.Module):
    # def __init__(self, hidden_dim, num_head, inner_dim):
    def __init__(self, d_model, d_inner, n_head, d_key, d_value, dropout=0.1):
        super().__init__()

        self.MultiHeadAttention = MultiHeadAttention(n_head, d_model, d_key, d_value, dropout=dropout)
        self.PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, input, mask=None):
        '''
        input shape : [batch size, channel, height, width]
                    : [128, 512, 4, 4]
        '''
        output, _ = self.MultiHeadAttention(input, input, input, mask=mask)
        # output = output.view(-1, 1, input.shape[1])
        output = self.PositionwiseFeedForward(output)
        # output = output + output

        return output


class PatchEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.bbox_token = torch.zeros(1)
        # self.segm_token = torch.zeros(1)
        self.conv = nn.Conv2d(512, d_model, kernel_size=1, stride=1)

    def forward(self, enc_srcs):
        batch_size = enc_srcs.shape[0]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = batch_size)

        outputs = self.conv(enc_srcs)
        outputs = outputs.reshape(batch_size, 16, -1)
        outputs = torch.cat([outputs, cls_tokens], dim=1)

        # bbox_token = self.bbox_token.cuda()
        # segm_token = self.segm_token.cuda()
        
        # bbox_token = bbox_token.repeat(batch_size, 1)
        # segm_token = segm_token.repeat(batch_size, 1)
        # outputs = torch.cat([bbox_token, enc_srcs, segm_token], dim = 1)

        return outputs


class Transformer(nn.Module):
    def __init__(self, net, n_head, d_key, d_value, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.resnet = net
        self.patch_embedding = PatchEmbedding(d_model)
        self.encoder = EncoderLayer(d_model, d_inner, n_head, d_key, d_value, dropout=dropout)
        # self.decoder = DecoderLayer()
        self.fc = nn.Linear(13056, 100)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, enc_srcs):

        # print('========== ResNet34 ==========')
        # print('ResNet34 Input : ', enc_srcs.shape)
        output = self.resnet(enc_srcs)
        # print('ResNet34 Output : ', output.shape)
        # print()
        
        # print('========== Patch Embedding ==========')
        # print('Patch Embedding Input : ', output.shape)
        output = self.patch_embedding(output)
        # print('Patch Embedding Output : ', output.shape)
        # print()

        # print('========== MLS ==========')
        # print('MLS input : ', output.shape)
        output = self.encoder(output)
        # print('MLS output : ', output.shape)
        # print()

        output = torch.flatten(output, 1)
        # print('output : ', output.shape)
        # print()

        output = self.fc(output).view(-1, 100)
        output = self.softmax(output)

        return output