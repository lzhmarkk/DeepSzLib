import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphLearner(nn.Module):
    def __init__(self, adj, dim, n_nodes, seq_len, dynamic=False):
        super().__init__()

        self.adj = adj
        self.dim = dim
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.dynamic = dynamic

        self.graph_construct_methods = {'attn': 0.0,
                                        'add': 0.0,
                                        'knn': 1.0}

        if self.dynamic:
            self.in_dim = self.n_nodes * self.dim
        else:
            self.in_dim = self.dim

        if 'attn' in self.graph_construct_methods:
            self.get_attn_graph_q = nn.Linear(self.in_dim, self.dim)
            self.get_attn_graph_k = nn.Linear(self.in_dim, self.dim)
        if 'add' in self.graph_construct_methods:
            self.get_add_graph_q = nn.Linear(self.in_dim, 1)
            self.get_add_graph_k = nn.Linear(self.in_dim, 1)
        if 'knn' in self.graph_construct_methods:
            self.k = 5

    def construct_graph(self, x):
        # (B, C, D)
        graph = []

        assert np.sum(list(self.graph_construct_methods.values())) == 1
        if 'attn' in self.graph_construct_methods and self.graph_construct_methods['attn'] > 0:
            q = self.get_attn_graph_q(x)
            k = self.get_attn_graph_k(x)
            adj_mx = torch.bmm(q, k.transpose(2, 1)) / np.sqrt(self.dim)
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

    def calculate_laplacian(self, adj):
        # (B, C, C)
        degree = torch.sum(adj, dim=2)
        # laplacian is sym or not
        adj = 0.5 * (adj + adj.transpose(2, 1))
        degree_l = torch.stack([torch.diag(degree[_]) for _ in range(len(degree))], dim=0)
        diagonal_degree_hat = torch.stack([torch.diag(1 / (torch.sqrt(degree[_]) + 1e-7)) for _ in range(len(degree))], dim=0)
        laplacian = torch.matmul(diagonal_degree_hat, torch.matmul(degree_l - adj, diagonal_degree_hat))
        return laplacian

    def forward(self, x):
        # (T, B, C, D)
        if self.dynamic:
            graphs = []
            for t in range(self.seq_len):
                g = self.construct_graph(x[t])
                g = self.calculate_laplacian(g)
                graphs.append(g)

            graphs = torch.stack(graphs, dim=0)  # (T, B, C, C)
            return graphs

        else:
            x = x.permute(1, 2, 0, 3).reshape(x.shape[1], self.n_nodes, -1)  # (B, C, T*D)
            graphs = self.construct_graph(x)
            # todo laplacian
            # graphs = self.calculate_laplacian(graphs)  # (B, C, C)
            graphs = 0.5 * (graphs + graphs.transpose(2, 1))
            return graphs
