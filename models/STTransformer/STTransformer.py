import torch
import numpy as np
import torch.nn as nn
from models.utils import Segmentation
from .Encoder import SpatialTemporalEncoder
from models.DCRNN.graph import distance_support
from .memory import MemoryNetwork


class STTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.window = args.window
        self.hidden = args.hidden
        self.layers = args.layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.dropout = args.dropout
        self.position_encoding = args.pos_enc
        self.preprocess = args.preprocess
        self.use_support = args.use_support
        self.filter_type = args.filter_type
        self.multi_task = args.multi_task

        self.init_func = args.init_func
        self.msg_method = args.msg_method
        self.upd_method = args.upd_method
        self.memory_activation = args.memory_activation

        # preprocess
        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        support = distance_support(self.channels) - np.eye(self.channels)
        self.adj = support

        self.memory_network = MemoryNetwork(self.hidden, self.channels, self.window // self.seg,
                                            self.init_func, self.msg_method, self.upd_method,
                                            self.memory_activation, self.dropout)

        self.encoder = SpatialTemporalEncoder(layers=self.layers, hidden=self.hidden, heads=self.heads, dropout=self.dropout,
                                              seq_len=1 + self.window // self.seg, n_channels=self.channels, filter_type=self.filter_type)

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.channels, self.hidden), requires_grad=True)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x)  # (B, T, C, D)

        # memory-network
        x = x.permute(1, 0, 2, 3)  # (T, B, C, D)
        x = self.memory_network(x)  # (T, B, C, D)

        x = torch.cat([self.cls_token.expand(-1, bs, -1, -1), x], dim=0)  # (1+T, B, C, D)

        # todo graph

        # x = x.permute(1, 0, 2, 3).reshape(1 + self.window // self.seg, bs * self.channels, self.hidden)  # (1+T, B*C, D)
        x = x.reshape(1 + self.window // self.seg, bs * self.channels, self.hidden)
        z = self.encoder(x, None)  # (1+T, B*C, D)

        # decoder
        z = z[0, :, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)

        z = torch.tanh(z)
        z = self.decoder(z).squeeze()  # (B)

        return z
