import torch
import torch.nn as nn
from .layer import MyEncoderLayer


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, layers, hidden, heads, dropout, seq_len, n_channels, filter_type, norm=False):
        super().__init__()

        self.n_layers = layers
        self.hidden = hidden
        self.heads = heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.norm = norm
        self.n_channels = n_channels
        self.filter_type = filter_type
        layer = MyEncoderLayer(d_model=self.hidden, nhead=self.heads, dropout=self.dropout,
                               nchannel=self.n_channels, filter_type=self.filter_type, seq_len=self.seq_len)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(layer)

        if self.norm:
            self.ln = nn.LayerNorm(self.hidden)

    def forward(self, x, graphs):
        h = [x]
        for mod in self.layers:
            x = mod(x, graphs)
            h.append(x)

        x = h[-1]

        if self.norm:
            x = self.ln(x)

        return x
