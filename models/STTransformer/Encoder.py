import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import MyEncoderLayer
from .graph import GNN


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, layers, hidden, heads, dropout, seq_len, n_channels,
                 gnn_layers, gnn_method, gnn_activation, norm=False):
        super().__init__()

        self.n_layers = layers
        self.hidden = hidden
        self.heads = heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.norm = norm
        self.n_channels = n_channels
        self.gnn_method = gnn_method
        self.gnn_layers = gnn_layers
        self.gnn_activation = gnn_activation

        # temporal
        layer = MyEncoderLayer(d_model=self.hidden, nhead=self.heads, dropout=self.dropout,
                               seq_len=self.seq_len)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(layer)

        # spatial
        self.drop_graph_eye = False
        self.conv = nn.ModuleList()
        self.eps = nn.ParameterList()
        for _ in range(self.gnn_layers):
            self.conv.append(GNN(self.hidden, self.n_channels, self.gnn_method, self.dropout, self.gnn_activation))
            self.eps.append(nn.Parameter(torch.randn(self.seq_len, 1, self.n_channels, self.hidden), requires_grad=True))

        if self.norm:
            self.ln = nn.LayerNorm(self.hidden)

    def spatial(self, x, graphs):
        # (T, B, C, D), ((T), B, C, C)
        h = []
        dynamic = graphs.ndim == 4

        for hop in range(self.gnn_layers):
            # drop eye
            if self.drop_graph_eye:
                if dynamic:
                    eye = self.eye.repeat(*graphs.shape[:2], 1, 1)
                else:
                    eye = self.eye[0].repeat(graphs.shape[0], 1, 1)
                eye = eye.bool().to(graphs.device)
                _graphs = graphs * (~eye)
            else:
                _graphs = graphs

            prop = self.conv[hop](x, _graphs)
            x = x + prop
            h.append(x)

        h = torch.cat(h, dim=-1)
        return h

    def forward(self, x, graphs):
        # (T, B*C, D), ((T), B, C, C)
        h = [x]

        x = self.layers[0](x, graphs)
        h.append(x)

        x = self.layers[1](x, graphs)
        h.append(x)

        if graphs is not None:
            x = x.reshape(self.seq_len, -1, self.n_channels, self.hidden)
            x = self.spatial(x, graphs)
            x = x.reshape(self.seq_len, -1, self.hidden)

        if self.norm:
            x = self.ln(x)

        return x
