import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphLearner(nn.Module):
    def __init__(self, adj, dim, n_nodes, seq_len, graph_construct_methods, dynamic=False, symmetric=True):
        super().__init__()

        self.adj = adj
        self.dim = dim
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.dynamic = dynamic
        self.symmetric = symmetric

        self.graph_construct_methods = graph_construct_methods

        if self.dynamic:
            self.in_dim = self.dim
        else:
            self.in_dim = self.seq_len * self.dim

        if 'attn' in self.graph_construct_methods and self.graph_construct_methods['attn'] > 0:
            self.get_attn_graph_q = nn.Linear(self.in_dim, self.dim)
            self.get_attn_graph_k = nn.Linear(self.in_dim, self.dim)
        if 'add' in self.graph_construct_methods and self.graph_construct_methods['add'] > 0:
            self.get_add_graph_q = nn.Linear(self.in_dim, 1)
            self.get_add_graph_k = nn.Linear(self.in_dim, 1)
        if 'knn' in self.graph_construct_methods and self.graph_construct_methods['knn'] > 0:
            self.k = 5

    def construct_graph(self, x):
        # (B, C, D)
        graph = []

        assert np.sum(list(self.graph_construct_methods.values())) == 1
        if 'attn' in self.graph_construct_methods and self.graph_construct_methods['attn'] > 0:
            q = self.get_attn_graph_q(x)
            k = self.get_attn_graph_k(x)
            adj_mx = torch.bmm(q, k.transpose(2, 1)) / np.sqrt(self.in_dim)
            adj_mx = torch.softmax(adj_mx, dim=-1)

            graph.append(adj_mx * self.graph_construct_methods['attn'])

        if 'add' in self.graph_construct_methods and self.graph_construct_methods['add'] > 0:
            q = self.get_add_graph_q(x)  # (B, C, 1)
            k = self.get_add_graph_k(x)
            adj_mx = q + k.transpose(2, 1)
            adj_mx = F.leaky_relu(adj_mx)
            adj_mx = torch.softmax(adj_mx, dim=-1)

            graph.append(adj_mx * self.graph_construct_methods['attn'])

        if 'knn' in self.graph_construct_methods and self.graph_construct_methods['knn'] > 0:
            norm = torch.norm(x, dim=-1, p="fro").unsqueeze(-1)
            x_norm = x / norm
            dist = torch.matmul(x_norm, x_norm.transpose(1, 2))
            knn_val, knn_ind = torch.topk(dist, self.k, dim=-1, largest=True)  # largest similarities

            adj_mx = (torch.ones_like(dist) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)
            adj_mx = F.leaky_relu(adj_mx)

            graph.append(adj_mx * self.graph_construct_methods['knn'])

        graph = torch.stack(graph, dim=0).sum(dim=0)  # (B, C, C)
        return graph

    def forward(self, x):
        # (T, B, C, D)
        if self.dynamic:
            graphs = []
            for t in range(self.seq_len):
                g = self.construct_graph(x[t])
                graphs.append(g)

            graphs = torch.stack(graphs, dim=0)  # (T, B, C, C)

        else:
            x = x.permute(1, 2, 0, 3).reshape(x.shape[1], self.n_nodes, -1)  # (B, C, T*D)
            graphs = self.construct_graph(x)  # (B, C, C)

        if self.symmetric:
            graphs = (graphs + graphs.transpose(-2, -1)) / 2

        return graphs


class GNN(nn.Module):
    def __init__(self, dim, n_nodes, method, dropout, activation):
        super().__init__()

        self.dim = dim
        self.n_nodes = n_nodes
        self.method = method
        self.dropout = dropout
        self.activation = activation

        if self.method == 'gcn':
            self.eye = torch.eye(self.n_nodes).bool()
            self.w = nn.Linear(self.dim, self.dim)
        elif self.method == 'sage':
            self.eye = torch.eye(self.n_nodes).bool()
            self.w1 = nn.Linear(self.dim, self.dim)
            self.w2 = nn.Linear(self.dim, self.dim)
        elif self.method == 'gat':
            self.w = nn.Linear(self.dim, self.dim)
            self.a1 = nn.Linear(self.dim, 1)
            self.a2 = nn.Linear(self.dim, 1)
            self.attn_drop = nn.Dropout(self.dropout)
            self.bias = nn.Parameter(torch.randn(self.dim), requires_grad=True)

    def forward(self, x, graph):
        # (..., N, D), (..., N, N) or (..., T, N, D), (..., T, N, N)

        if self.method == 'gcn':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(graph.device)
            graph = graph + eye
            D = 1 / (torch.sqrt(graph.sum(dim=-1)) + 1e-7)
            D = torch.diag_embed(D)  # (..., N, N)
            graph = torch.matmul(D, torch.matmul(graph, D))  # (..., N, N)
            x = torch.matmul(graph, x)
            x = self.w(x)

        elif self.method == 'sage':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(graph.device)
            graph = graph * (~eye)
            graph = graph / (torch.sum(graph, dim=-1, keepdim=True) + 1e-7)  # (..., N, N)
            neighbor = torch.matmul(graph, x)  # (..., N, N)
            x = self.w1(x) + self.w2(neighbor)

        elif self.method == 'gat':
            x = self.w(x)  # (..., N, D)
            a = self.a1(x) + self.a2(x).transpose(-2, -1)  # (..., N, N)
            a = F.leaky_relu(a)  # (..., N, N)
            a = torch.softmax(a, dim=-1)  # (..., N, N)
            a = self.attn_drop(a)
            x = torch.matmul(a, x) + self.bias

        if self.activation == 'relu':
            x = torch.relu(x)
        elif self.activation == 'leaky':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)

        return x
