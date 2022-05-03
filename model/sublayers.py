import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import ScaledDotProductAttention, SigmoidScaledDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_key, d_value, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        # self.w_query = nn.Linear(d_model, n_head * d_key)
        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(temperature = d_key ** 0.5)
        self.attention_sigmoid = SigmoidScaledDotProductAttention(temperature = d_key ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)        

    def forward(self, query, key, value, mask=None):

        d_key, d_value, n_head = self.d_key, self.d_value, self.n_head
        # size_batch, len_query, len_key, len_value = query.size(0), query.size(1), key.size(1), value.size(1)
        size_batch = query.shape[0]
        # print(query.size(0))

        len_query = 1
        len_key = 1
        len_value = 1

        residual = query

        print(query.shape)
        query = self.w_query(query).view(size_batch, len_query, n_head, d_key)
        key = self.w_key(key).view(size_batch, len_key, n_head, d_key)
        value = self.w_value(value).view(size_batch, len_value, n_head, d_value)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        
        query, attn = self.attention(query, key, value, mask=mask)

        query = query.transpose(1, 2).contiguous().view(size_batch, len_query, -1)
        query = self.dropout(self.fc(query))
        query += residual.reshape(-1, 1, self.d_model)
        # query += residual.reshape(-1, batch_size, )


        query = self.layer_norm(query)

        return query, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x