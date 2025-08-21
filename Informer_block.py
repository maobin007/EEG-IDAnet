import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from math import sqrt
from einops import rearrange
from utils.masking import ProbMask, TriangularCausalMask


# position embedding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=150, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        # Initialize a Position Encoding
        # d_model: Embedding dimensions
        # max_seq_len:Maximum sequence length
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

    def forward(self, x): # x->[batch, embed_dim, length]
        # x = x + self.encoding[:, :x.shape[1], :].cuda()
        # print(self.encoding[:, :x.shape[1], :].shape)
        # print(x.shape)
        x = x + self.encoding[:, :x.shape[1], :]
        return self.dropout(x)

class VanillaAttention(nn.Module):
    """vanilla attention"""
    def __init__(self, use_mask=False, dropout=0.1, factor=5):
        super(VanillaAttention, self).__init__()
        self.use_mask = use_mask
        self.dropout = nn.Dropout(dropout)
        self.factor = factor
        self.attn = None

    def forward(self, q, k, v, mask=None):
        attn = torch.einsum("nhid, nhjd -> nhij", q, k) * (k.shape[-1] ** -0.5)
        if self.use_mask:
            if mask is None:
                shape = (attn.shape[0], 1, attn.shape[2], attn.shape[3])
                # torch.triu sets elements below the main diagonal to False, and elements above the main diagonal to True.
                mask = torch.triu(torch.ones(shape, dtype=torch.bool), diagonal=1).to(attn.device)
            # based on the boolean value of mask, the positions in the attn tensor corresponding to True are filled with -∞
            attn.masked_fill_(mask, -np.inf)
        attn = torch.softmax(attn, dim=-1)
        self.attn = self.dropout(attn)

        out = torch.einsum("nhij, nhjd -> nhid", attn, v)
        return out, attn

class ProbAttention(nn.Module):
    def __init__(self, use_mask=True, dropout=0.1, factor=5):
        super(ProbAttention, self).__init__()
        self.use_mask = use_mask
        self.dropout = nn.Dropout(dropout)
        self.factor = factor
    def forward(self, q, k, v, mask=None):
        N, H, q_len, D = q.shape
        _, _, k_len, _ = k.shape
        _, _, v_len, _ = v.shape
        # 1.compute u and U, num_q represents u, num_k represents U
        num_q,  num_k = [int(self.factor * np.ceil(np.log(length))) for length in [q_len, k_len]]
        # print(np.ceil(np.log(q_len))) #5.0
        # print('num_q', num_q)
        num_q, num_k = np.minimum(num_q, q_len), np.minimum(num_k, k_len)
        # 2.randomly selecting a small number of K
        k_expanded = k.unsqueeze(-3).expand(-1, -1, q_len, -1, -1)
        random_index = torch.randint(k_len, size=(q_len, num_k))
        k_sampled = k_expanded[:, :, torch.arange(q_len).unsqueeze(1), random_index, :]
        # print('q',q.shape)
        # print(k_sampled.shape)
        # 3.compute pre_attention
        pre_attn = torch.einsum("bhid, bhijd->bhij", q, k_sampled)
        # 4.get the M used to select a small number of q_i
        measure = pre_attn.max(-1)[0] - pre_attn.sum(-1)/k_len
        # 5.select a small number of q_i
        q_selected_index = measure.topk(num_q, sorted=False)[1]
        q_selected = q.gather(-2, q_selected_index.unsqueeze(-1).expand(-1, -1, -1, q.shape[-1]))
        # 6.compute attention
        attn = torch.einsum('bhid, bhjd->bhij', q_selected, k) * (k.size(-1) ** -0.5)
        # print(attn.shape)
        # add mask and get the baseboard information output by contex
        if self.use_mask:
            assert  q_len == v_len
            mask = ProbMask(v.shape[0], v.shape[1], q_len, q_selected_index, attn)
            attn.masked_fill_(mask.mask, -np.inf)

            # setting uniform information as a contex
            context = v.cumsum(dim=-2) # 7
        else:
            v_mean = v.mean(dim=-2)
            context = v_mean.unsqueeze(-2).expand(-1, -1, q_len, v_mean.shape[-1]).clone() #7
        attn = torch.softmax(attn, dim=-1)
        self.attn = self.dropout(attn)
        # 8.obtaining self-attention results
        out = torch.einsum("bhij, bhjd->bhid", self.attn, v)
        q_selected_index = q_selected_index.unsqueeze(-1)
        # restore the output value
        out_scatter_index = q_selected_index.expand(-1, -1, -1, out.shape[-1])
        out = context.scatter(2, out_scatter_index, out)
        return out,attn

# Multi-heand Prob self-attention
class AttentionBlock(nn.Module):
    def __init__(self, attn_type, d_model, num_heads, use_mask=False, dropout=0.1, factor=5):
        super(AttentionBlock, self).__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.num_heads = num_heads

        assert attn_type in ['vanilla', 'prob']
        if attn_type == 'vanilla':
            self.attention = VanillaAttention(use_mask=use_mask, dropout=dropout, factor=factor)
        else:
            self.attention = ProbAttention(use_mask=use_mask, dropout=dropout, factor=factor)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, q, k, v, mask=None):
        q_, k_, v_ = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q_, k_, v_ = map(lambda item: rearrange(item, "N L (H d) -> N H L d", H=self.num_heads), (q_, k_, v_))
        out, attn_weight = self.attention(q_, k_, v_, mask)
        out = rearrange(out, 'N H L D -> N L (H D)')
        return self.norm(out + q), attn_weight

class MLPBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, act='relu'):
        super(MLPBlock, self).__init__()
        d_ff = d_ff * d_model or d_model * 4
        act = nn.ReLU() if act=='relu' else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.net(x)
        return self.norm(out+x)

class ConvBlock(nn.Module):
    def __init__(self, cin, kerSize):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cin, kernel_size=kerSize,stride=1,padding=(kerSize-1)//2, bias=False),
            nn.BatchNorm2d(cin),
            nn.ELU(),
            nn.MaxPool1d(3, 2, 1)
        )
    def forward(self, x):
        """
        :param x:  (N, L, D)
        :return:
        """
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)

class InformerEncoder_Layer(nn.Module):
    def __init__(self, attn_type, d_model, num_heads, d_ff, use_conv=False, attn_dropout=0.1, ddropout=0.1,factor=5, act='relu'):
        super(InformerEncoder_Layer, self).__init__()
        self.attention = AttentionBlock(attn_type=attn_type, d_model=d_model, num_heads=num_heads, use_mask=False, dropout=attn_dropout, factor=factor)
        self.mlp = MLPBlock(d_model=d_model, d_ff=d_ff, dropout=ddropout, act=act)
        if use_conv:
            self.conv = ConvBlock(d_model)
        else:
            self.conv = nn.Identity()
    def forward(self, x, mask=None):
        out, attn_weight = self.attention(x, x, x, mask)
        out = self.mlp(out)
        out = self.conv(out)
        return out, attn_weight

class InformerEncoder(nn.Module):
    def __init__(self, num_layers,attn_type, d_model, num_heads, d_ff, use_conv=False, attn_dropout=0.1, ddropout=0.1, factor=5,
                 act='relu'):
        super(InformerEncoder, self).__init__()
        # attention+mlp+conv
        self.position = PositionalEncoding(d_model, dropout=0.1)
        self.layers = nn.ModuleList(
            [InformerEncoder_Layer(
                attn_type, d_model, num_heads, d_ff, use_conv, attn_dropout, ddropout, factor, act
            ) for _ in range(num_layers-1)]
        )
        self.layers.append(InformerEncoder_Layer(attn_type, d_model, num_heads, d_ff, use_conv, attn_dropout, ddropout, factor, act))
    def forward(self, x, mask=None):
        x = self.position(x)
        for layer in self.layers:
            x, attn_weight = layer(x, mask)
        return x, attn_weight
#
# #
# ###============================ Initialization parameters ============================###
# channels        = 64
# samples         = 15
#
# ###============================ main function ============================###
# def main():
#     input = torch.randn(32, channels, samples)
#     model = InformerEncoder(num_layers=1,attn_type='prob', d_model=15, num_heads=5, d_ff=2, use_conv=False, attn_dropout=0.1, ddropout=0.1, factor=5,
#                  act='relu')
#     out, attn_weight = model(input)
#     print('===============================================================')
#     print('out', out.shape)
#     print('attn_weight',attn_weight.shape)
#     print('model', model)
#     # summary(model=model, input_size=(channels,samples), device="cpu")
#     # stat(model, (1, channels, samples))
#
# if __name__ == "__main__":
#     main()















