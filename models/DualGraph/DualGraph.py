import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Segmentation
from .graphLearner import LocalGraphLearner
from .conv import LocalGNN


class DualGraph(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden = args.hidden
        self.n_channels = args.n_channels
        self.seq_len = args.window // args.seg
        self.dropout = args.dropout
        self.preprocess = args.preprocess

        self.local_knn = args.local_graph_knn
        self.local_graph_method = args.local_graph_method
        self.local_gnn_layers = args.local_gnn_layers
        self.local_gnn_method = args.local_gnn_method
        self.local_gnn_activation = args.local_gnn_activation
        self.local_separate_diag = args.local_separate_diag

        self.use_ffn = args.use_ffn

        if self.preprocess == 'seg':
            self.input_dim = args.input_dim
            self.segmentation = Segmentation(self.seg, self.input_dim, self.channels)
        elif self.preprocess == 'fft':
            self.input_dim = args.input_dim
            self.fc = nn.Linear(self.input_dim, self.hidden)

        # local
        self.local_graph_learner = nn.ModuleList()
        self.local_gnn = nn.ModuleList()
        self.local_ln = nn.ModuleList()
        for _ in range(self.local_gnn_layers):
            self.local_graph_learner.append(LocalGraphLearner(self.hidden, self.seq_len, self.local_graph_method, knn=self.local_knn, pos_enc=True))
            self.local_gnn.append(LocalGNN(self.hidden, self.seq_len, self.local_gnn_layers, self.dropout, self.local_gnn_method,
                                           self.local_gnn_activation, self.local_separate_diag))
            self.local_ln.append(nn.LayerNorm(self.hidden))

        # ffn
        if self.use_ffn:
            self.ffn = nn.Sequential(nn.Linear(self.hidden, 4 * self.hidden),
                                     nn.GELU(),
                                     nn.Linear(4 * self.hidden, self.hidden))
            self.ffn_ln = nn.LayerNorm(self.hidden)

        # decoder
        self.decoder = nn.Sequential(nn.Linear(self.n_channels * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x)  # (B, T, C, D)

        # local graph
        x = x.transpose(2, 1).reshape(bs * self.n_channels, self.seq_len, self.hidden)  # (B*C, T, D)
        for layer in range(self.local_gnn_layers):
            local_graph = self.local_graph_learner[layer](x)  # (B*C, T, T)
            x = self.local_gnn[layer](x, local_graph)  # (B*C, T, D)
            x = self.local_ln[layer](x)  # (B*C, T, D)

        # ffn
        z = x
        if self.use_ffn:
            z = self.ffn_ln(z + self.ffn(z))

        # decoder
        z = torch.mean(z, dim=1)
        z = torch.tanh(z)
        z = z.reshape(bs, self.n_channels * self.hidden)  # (B, C, D)
        z = self.decoder(z).squeeze(dim=-1)  # (B)
        return z
