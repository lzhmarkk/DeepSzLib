import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F


class MyEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, nchannel, filter_type, seq_len):
        super().__init__()
        self.d_model = d_model
        self.ffn_hidden = 4 * d_model
        self.nhead = nhead
        self.dropout = dropout
        self.nchannel = nchannel
        self.seq_len = seq_len
        self.filter_type = filter_type

        # temporal parameters
        self.mode = 'attn'
        if self.mode == 'attn':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif self.mode == 'fc':
            self.fc = nn.Conv2d(seq_len, seq_len, kernel_size=1)
        elif self.mode == 'mlp':
            self.mlp = nn.Sequential(nn.Conv2d(seq_len, 4 * (seq_len - 1), kernel_size=1),
                                     nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(4 * (seq_len - 1), seq_len, kernel_size=1))
        else:
            raise ValueError()

        # spatial parameters
        self.spatial_linear = nn.Linear(d_model * (2 if filter_type == 'dual_random_walk' else 1), d_model, bias=False)

        # ffn parameters
        self.ffn = nn.Sequential(nn.Linear(d_model, self.ffn_hidden),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.ffn_hidden, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def temporal(self, x):
        # todo 考虑特征波形的特征
        if self.mode == 'attn':
            x = self.self_attn(x, x, x, need_weights=False)[0]
        elif self.mode == 'fc':
            x = self.fc(x)
        elif self.mode == 'mlp':
            x = self.mlp(x)
        else:
            raise ValueError()
        return x

    def spatial(self, x, graphs):
        # todo 考虑半球的对称性特征
        x = x.reshape(self.seq_len, -1, self.nchannel, self.d_model)
        h = []
        for g in graphs:
            h.append(torch.einsum("tbcd, cn->tbnd", x, g))
        h = self.spatial_linear(torch.cat(h, dim=-1))
        return h.reshape(self.seq_len, -1, self.d_model)

    def forward(self, x: Tensor, graphs: Tensor):
        # (T, B*C, D)
        x_enc = self.temporal(x)
        x = self.norm1(x + self.dropout1(x_enc))

        # x_enc = self.spatial(x, graphs)
        # x = self.norm3(x + self.dropout3(x_enc))

        x_enc = self.ffn(x)
        x = self.norm2(x + self.dropout2(x_enc))

        return x
