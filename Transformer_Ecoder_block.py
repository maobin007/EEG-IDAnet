import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce #对张量的重塑和维度变换
from einops import rearrange
from torchinfo import summary

# position embedding module
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        # 初始化一个Position Encoding
        # embed_dim:Embedding dimensions
        # max_seq_len:Maximum sequence length
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

    def forward(self, x): # x->[batch, embed_dim, length]
        x = x + self.encoding[:, :x.shape[1], :]
        # print(x.shape)
        # print(self.encoding[:,:x.shape[1],:].shape)
        # x = x + self.encoding[:, :x.shape[1], :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.act = nn.ELU()

    def forward(self, x: Tensor, mask: Tensor=None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)
        # print(keys.shape) # [32, 4, 15, 8]
        # print(self.values(x).shape)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy/scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(nn.Linear(emb_size, expansion*emb_size, bias=False),
                         nn.GELU(),
                         nn.Dropout(drop_p),
                         nn.Linear(expansion*emb_size, emb_size))


class ReidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.1, forward_expansion=4, forward_drop_p=0.1):
        super().__init__(
            ReidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
            ),emb_size, drop_p),
            ReidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
            ), emb_size, drop_p)
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads=heads) for _ in range(depth)])


class TransformEncoder_block(nn.Module):
    def __init__(self, emb_size, heads, depth):
        super(TransformEncoder_block, self).__init__()
        self.position = PositionalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)

    def forward(self, x):
        x = self.position(x)
        # print(x.shape)
        x = self.trans(x)
        return x

