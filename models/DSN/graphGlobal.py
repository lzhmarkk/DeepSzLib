import math
import torch
import torch.nn as nn
from models.DCRNN.graph import distance_support
from models.MTGNN.MTGNN import graph_constructor


class GlobalGraphLearner(nn.ModuleList):
    def __init__(self, dim, subgraph_nodes, n_subgraphs, method, dropout, channels, pos_enc):
        super().__init__()

        self.dim = dim
        self.subgraph_nodes = subgraph_nodes
        self.n_subgraphs = n_subgraphs
        self.method = method
        self.pos_enc = pos_enc
        self.dropout = dropout

        if self.method == 'predefined':
            assert self.subgraph_nodes == 1
            self.adj_mx = torch.from_numpy(distance_support(channels)).float()
        elif self.method == 'attn':
            self.w_q = nn.Linear(self.dim, self.dim)
            self.w_k = nn.Linear(self.dim, self.dim)
        elif self.method == 'fc':
            self.w = nn.Sequential(nn.Linear(2 * self.dim, self.dim),
                                   nn.ReLU(),
                                   nn.Linear(self.dim, 1))
        elif self.method == 'add':
            self.w = nn.Linear(self.dim, 1)
        elif self.method == 'cosine':
            pass
        elif self.method == 'emb':
            self.gc = graph_constructor(n_subgraphs, dim)

        if self.pos_enc:
            self.pos_emb = nn.Embedding(self.n_subgraphs, self.dim)

    def forward(self, x):
        # (B, C, N, D)
        bs = x.shape[0]

        if self.pos_enc:
            pos_emb = self.pos_emb.weight.reshape(1, self.n_subgraphs, 1, self.dim)  # (1, C, 1, D)
            x = x + pos_emb

        x = x.reshape(bs, self.n_subgraphs * self.subgraph_nodes, self.dim)  # (B, C*N, D)

        # build global graph
        if self.method == 'predefined':
            adj_mx = self.adj_mx.to(x.device)

        elif self.method == 'attn':
            q = self.w_q(x)
            k = self.w_k(x)
            adj_mx = torch.bmm(q, k.transpose(2, 1)) / math.sqrt(self.dim)
            adj_mx = torch.softmax(adj_mx, dim=-1)

        elif self.method == 'fc':
            q = x.unsqueeze(dim=1).repeat(1, self.n_subgraphs * self.subgraph_nodes, 1, 1)
            k = x.unsqueeze(dim=2).repeat(1, 1, self.n_subgraphs * self.subgraph_nodes, 1)
            qk = torch.cat([q, k], dim=-1)
            adj_mx = self.w(qk).squeeze(dim=-1)
            adj_mx = torch.sigmoid(adj_mx)

        elif self.method == 'add':
            q = x.unsqueeze(dim=1).repeat(1, self.n_subgraphs * self.subgraph_nodes, 1, 1)
            k = x.unsqueeze(dim=2).repeat(1, 1, self.n_subgraphs * self.subgraph_nodes, 1)
            adj_mx = self.w(q + k).squeeze(dim=-1)
            adj_mx = torch.sigmoid(adj_mx)

        elif self.method == 'cosine':
            norm = torch.norm(x, dim=-1, p="fro", keepdim=True)
            x_norm = x / (norm + 1e-7)
            adj_mx = torch.bmm(x_norm, x_norm.transpose(1, 2))
            adj_mx = torch.relu(adj_mx)

        elif self.method == 'emb':
            adj_mx = self.gc(x).unsqueeze(dim=0).repeat(x.shape[0], 1, 1)

        else:
            adj_mx = torch.eye(self.n_subgraphs * self.subgraph_nodes, self.subgraph_nodes * self.n_subgraphs)
            adj_mx = adj_mx.unsqueeze(dim=0).repeat(bs, 1, 1).to(x.device)

        return adj_mx  # (B, C*N, N*C)
