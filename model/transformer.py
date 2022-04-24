# import math
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.nn import CrossEntropyLoss
# from model.util import clones
# from transformers.activations import 
# import numpy as np

# """
# Positional Encoding
# Encoder
# Decoder
# Transformer
# """




# # """
# # self-Attention의 경우 Query, Key, Value를 입력으로 받아
# # MatMul(Q, K) -> Scale -> Masking(opt. Decoder) -> Softmax -> MatMul(result, V)

# # """

# # def self_attention(query, key, value, mask=None):
# #     key_transpose = torch.transpose(key, -2, -1)
# #     matmul_result = torch.matmul(query, key_transpose)
# #     d_k = query.size()[-1]
# #     attention_score = matmul_result / math.sqrt(d_k)

# #     if mask is not None:
# #         attention_score = attention_score.masked_fill(mask == 0, -1e20)

# #     softmax_attention_score = F.softmax(attention_score, dim=-1)
# #     result = torch.matmul(softmax_attention_score, value)

# #     return result, softmax_attention_score

# # class MultiHeadAttention(nn.Module):
# #     # init
# #     # forward 

# # class FeedForward