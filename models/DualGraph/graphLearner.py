import torch
import torch.nn as nn
import numpy as np


class LocalGraphLearner(nn.Module):
    def __init__(self, dim, n_nodes, method, knn=None, pos_enc=True):
        super().__init__()

        self.dim = dim
        self.n_nodes = n_nodes
        self.method = method
        self.knn = knn
        self.pos_enc = pos_enc

        if self.method == 'attn':
            self.w_q = nn.Linear(self.dim, self.dim)
            self.w_k = nn.Linear(self.dim, self.dim)
        elif self.method == 'cosine':
            pass

        if self.pos_enc:
            self.pos_emb = nn.Parameter(torch.randn([self.n_nodes, self.dim]), requires_grad=True)

    def forward(self, x):
        # (B, N, D)
        if self.pos_enc:
            x = x + self.pos_emb.unsqueeze(dim=0)

        if self.method == 'attn':
            q = self.w_q(x)
            k = self.w_k(x)
            adj_mx = torch.bmm(q, k.transpose(2, 1)) / np.sqrt(self.dim)

            adj_mx = torch.softmax(adj_mx, dim=-1)

        elif self.method == 'cosine':
            norm = torch.norm(x, dim=-1, p="fro", keepdim=True)
            x_norm = x / (norm + 1e-7)
            adj_mx = torch.bmm(x_norm, x_norm.transpose(1, 2))

        else:
            raise ValueError()

        # select k-largest values
        if self.knn is not None and self.knn > 0:
            knn_val, knn_ind = torch.topk(adj_mx, self.knn, dim=-1, largest=True)
            adj_mx = (torch.ones_like(adj_mx) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)

        return adj_mx  # (N, N)
