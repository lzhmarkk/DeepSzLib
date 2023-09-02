import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Segmentation
from .graphLocal import LocalGraphLearner
from .graphGlobal import GlobalGraphLearner
from .conv import LocalGNN, GlobalGNN
from .pooling import Pooling


class DualGraph(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden = args.hidden
        self.n_channels = args.n_channels
        self.seq_len = args.window // args.seg
        self.dropout = args.dropout
        self.preprocess = args.preprocess
        self.dataset = args.dataset

        self.local_knn = args.local_graph_knn
        self.local_graph_method = args.local_graph_method
        self.local_gnn_layers = args.local_gnn_layers
        self.local_gnn_method = args.local_gnn_method
        self.local_gnn_activation = args.local_gnn_activation
        self.local_gnn_depth = args.local_gnn_depth
        self.local_gnn_decay = args.local_gnn_decay

        self.pool_method = args.pool_method
        self.pool_heads = args.pool_heads
        self.pool_proxies = args.pool_proxies

        self.global_graph_method = args.global_graph_method
        self.global_gnn_layers = args.global_gnn_layers
        self.global_gnn_method = args.global_gnn_method
        self.global_gnn_activation = args.global_gnn_activation
        self.global_gnn_depth = args.global_gnn_depth

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
                                           self.local_gnn_activation, self.local_gnn_depth, self.local_gnn_decay))
            self.local_ln.append(nn.LayerNorm(self.hidden))

        # pooling
        self.pooling = Pooling(self.hidden, self.seq_len, self.n_channels, self.pool_method, self.pool_heads, self.pool_proxies)
        self.seq_len_pooled = self.pooling.subgraph_nodes_agg  # T'

        # global
        self.global_graph_learner = nn.ModuleList()
        self.global_gnn = nn.ModuleList()
        self.global_ln = nn.ModuleList()
        for _ in range(self.global_gnn_layers):
            self.global_graph_learner.append(GlobalGraphLearner(self.hidden, self.seq_len_pooled, self.n_channels,
                                                                self.global_graph_method, self.dropout, pos_enc=True))
            self.global_gnn.append(GlobalGNN(self.hidden, self.n_channels * self.seq_len_pooled, self.global_gnn_layers, self.dropout,
                                             self.global_gnn_method, self.global_gnn_activation, self.global_gnn_depth))
            self.global_ln.append(nn.LayerNorm(self.hidden))

        # ffn
        if self.use_ffn:
            self.ffn = nn.Sequential(nn.Linear(self.hidden, 4 * self.hidden),
                                     nn.GELU(),
                                     nn.Linear(4 * self.hidden, self.hidden))
            self.ffn_ln = nn.LayerNorm(self.hidden)

        # decoder
        self.activation = nn.Tanh() if self.dataset == 'FDUSZ' else nn.ReLU()
        self.channel_pooling = 'fc' if self.dataset == 'FDUSZ' else 'max'
        if self.channel_pooling == 'fc':
            self.decoder = nn.Sequential(nn.Linear(self.n_channels * self.hidden, self.hidden),
                                         nn.GELU(),
                                         nn.Linear(self.hidden, 1))
        else:
            self.decoder = nn.Linear(self.hidden, 1)

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

        x = x.reshape(bs, self.n_channels, self.seq_len, self.hidden)

        # local graph pooling
        x = self.pooling(x)  # (B, C, T', D)

        # global graph
        for layer in range(self.global_gnn_layers):
            global_graph = self.global_graph_learner[layer](x)  # (B, C*T', T'*C)
            x = x.reshape(bs, self.n_channels * self.seq_len_pooled, self.hidden)  # (B, C*T', D)
            x = self.global_gnn[layer](x, global_graph)  # (B, C*T', D)
            x = x.reshape(bs, self.n_channels, self.seq_len_pooled, self.hidden)  # (B, C, T', D)
            x = self.global_ln[layer](x)  # (B, C, T', D)

        # ffn
        z = x
        if self.use_ffn:
            z = self.ffn_ln(z + self.ffn(z))

        # decoder
        z = torch.mean(z, dim=-2)
        z = self.activation(z)

        if self.channel_pooling == 'fc':
            z = z.reshape(bs, self.n_channels * self.hidden)  # (B, C, D)
            z = self.decoder(z).squeeze(dim=-1)  # (B)
        else:
            z = z.reshape(bs, self.n_channels, self.hidden)  # (B, C, D)
            z = self.decoder(z).squeeze(dim=-1)  # (B, C)
            z, _ = torch.max(z, dim=1)

        return z, None
