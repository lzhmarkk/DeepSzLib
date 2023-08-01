import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import MyEncoderLayer


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, layers, hidden, heads, dropout, seq_len, n_channels, gnn_layers, filter_type, norm=False):
        super().__init__()

        self.n_layers = layers
        self.hidden = hidden
        self.heads = heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.norm = norm
        self.n_channels = n_channels
        self.filter_type = filter_type
        self.gnn_layers = gnn_layers

        # temporal
        layer = MyEncoderLayer(d_model=self.hidden, nhead=self.heads, dropout=self.dropout,
                               filter_type=self.filter_type, seq_len=self.seq_len)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(layer)

        # spatial
        self.spatial_linear = nn.Linear(self.gnn_layers * self.hidden, self.hidden)
        self.eps = nn.Parameter(torch.randn(self.seq_len, 1, self.n_channels, self.hidden), requires_grad=True)
        self.drop_graph_eye = False

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

            # conv
            if dynamic:
                prop = torch.einsum("tbmn,tbnd->tbmd", _graphs, x)
            else:
                prop = torch.einsum("bmn,tbnd->tbmd", _graphs, x)

            x = x + self.spatial_linear(prop)
            # todo norm
            # todo activation
            h.append(x)

            if dynamic:
                graphs = torch.einsum("tbmn,tbnm->tbmm", graphs, graphs)
            else:
                graphs = torch.bmm(graphs, graphs)

        h = torch.cat(h, dim=-1)

        gnn_act = 'none'
        if gnn_act == 'tanh':
            h = torch.tanh(h)
        elif gnn_act == 'relu':
            h = torch.relu(h)
        elif gnn_act == 'leaky':
            h = F.leaky_relu(h)

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
