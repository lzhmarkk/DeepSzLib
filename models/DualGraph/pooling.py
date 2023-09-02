import torch
import torch.nn as nn


class Pooling(nn.Module):
    def __init__(self, dim, n_nodes, n_graphs, pool_method, n_heads, n_proxy):
        super().__init__()

        self.dim = dim
        self.n_nodes = n_nodes
        self.n_graphs = n_graphs
        self.pool_method = pool_method
        self.n_proxy = n_proxy
        self.n_heads = n_heads

        if self.pool_method == 'mean':
            self.subgraph_nodes_agg = 1
        elif self.pool_method == 'max':
            self.subgraph_nodes_agg = 1
        elif self.pool_method == 'last':
            self.subgraph_nodes_agg = 1
        elif self.pool_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(self.n_graphs, 1, self.dim), requires_grad=True)
            self.aggregate = nn.MultiheadAttention(self.dim, self.n_heads, 0, batch_first=True)
            self.subgraph_nodes_agg = 1
        elif self.pool_method == 'proxy':
            self.cls_token = nn.Parameter(torch.randn(1, self.n_graphs, self.n_proxy, self.dim), requires_grad=True)
            self.aggregate = nn.MultiheadAttention(self.dim, self.n_heads, 0, batch_first=True)
            self.subgraph_nodes_agg = self.n_proxy
        else:
            self.subgraph_nodes_agg = self.n_nodes

    def forward(self, x):
        # (B, C, N, D)
        bs = x.shape[0]

        if self.pool_method == 'mean':
            x = x.mean(dim=-2, keepdims=True)  # (B, C, 1, D)
        elif self.pool_method == 'max':
            x = x.max(dim=-2, keepdims=True)[0]  # (B, C, 1, D)
        elif self.pool_method == 'last':
            x = x[:, :, [-1], :]  # (B, C, 1, D)
        elif self.pool_method == 'cls':
            x = x.reshape(bs * self.n_graphs, self.n_nodes, self.dim)  # (B*C, N, D)
            cls = self.cls_token.repeat(bs, 1, 1)  # (B*C, 1, D)
            x = self.aggregate(cls, x, x, need_weights=False)[0]  # (B*C, 1, D)
            x = x.reshape(bs, self.n_graphs, 1, self.dim)  # (B, C, 1, D)
        elif self.pool_method == 'proxy':
            x = x.reshape(bs * self.n_graphs, self.n_nodes, self.dim)  # (B*C, N, D)
            cls = self.cls_token.repeat(bs, 1, 1, 1).reshape(bs * self.n_graphs, self.n_proxy, self.dim)  # (B*C, P, D)
            x = self.aggregate(cls, x, x, need_weights=False)[0]  # (B*C, P, D)
            x = x.reshape(bs, self.n_graphs, self.n_proxy, self.dim)  # (B, C, P, D)

        return x  # (B, C, N', D)
