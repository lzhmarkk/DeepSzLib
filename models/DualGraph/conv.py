import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGNN(nn.Module):
    def __init__(self, dim, n_nodes, n_layers, dropout, method, activation, separate_diag):
        super().__init__()

        self.dim = dim
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

        self.method = method
        if self.method == 'gcn':
            self.w = nn.Linear(self.dim, self.dim)
        elif self.method == 'sage':
            self.w1 = nn.Linear(self.dim, self.dim)
            self.w2 = nn.Linear(self.dim, self.dim)
        elif self.method == 'gat':
            self.a1 = nn.Linear(self.dim, 1)
            self.a2 = nn.Linear(self.dim, 1)
            self.w = nn.Linear(self.dim, self.dim)
        elif self.method == 'rnn':
            self.i_r = nn.Linear(self.dim, self.dim)
            self.h_r = nn.Linear(self.dim, self.dim)
            self.i_z = nn.Linear(self.dim, self.dim)
            self.h_z = nn.Linear(self.dim, self.dim)
            self.i_m = nn.Linear(self.dim, self.dim)
            self.h_m = nn.Linear(self.dim, self.dim)
            self.separate_diag = separate_diag

        self.dropout = nn.Dropout(dropout)
        self.eye = torch.eye(n_nodes).bool()

    def forward(self, x, graph):
        # (..., N, D), (..., N, N)

        if self.method == 'gcn':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)
            graph = graph + eye
            D = 1 / (torch.sqrt(graph.sum(dim=-1)) + 1e-7)
            D = torch.diag_embed(D)  # (..., N, N)
            graph = torch.matmul(D, torch.matmul(graph, D))  # (..., N, N)
            graph = self.dropout(graph)
            x = torch.matmul(graph, x)
            x = self.w(x)

        elif self.method == 'sage':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)
            graph = graph * (~eye)  # Drop self-connection
            graph = graph / (torch.sum(graph, dim=-1, keepdim=True) + 1e-7)  # (..., N, N)
            graph = self.dropout(graph)
            neighbor = torch.matmul(graph, x)  # (..., N, N)
            x = self.w1(x) + self.w2(neighbor)

        elif self.method == 'gat':
            adj = F.leaky_relu(self.a1(x) + self.a2(x).transpose(-2, -1))  # (..., N, N)
            graph = self.dropout(adj) * graph
            graph = torch.softmax(graph, dim=-1)  # (..., N, N)
            x = torch.matmul(graph, x)
            x = self.w(x)  # (..., N, D)

        elif self.method == 'rnn':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)

            if self.separate_diag:
                graph = graph * (~eye)  # Drop self-connection
                diag = graph * eye
                x_shifted = F.pad(x, (0, 0, 1, 0))[..., :-1, :]  # (..., N, D)
                r = torch.sigmoid(torch.matmul(graph, self.i_r(x)) + torch.matmul(diag, self.h_r(x)))  # (..., N, D)
                z = torch.sigmoid(torch.matmul(graph, self.i_z(x)) + torch.matmul(diag, self.h_z(x)))  # (..., N, D)
                m = torch.matmul(graph, self.i_m(x))  # (..., N, D)
                m = torch.tanh(m + r * self.h_m(x_shifted))
                m = self.dropout(m)
                x = (1 - z) * m + z * x

            else:
                x_shifted = F.pad(x, (0, 0, 1, 0))[..., :-1, :]  # (..., N, D)
                r = torch.sigmoid(torch.matmul(graph, self.i_r(x)))  # (..., N, D)
                z = torch.sigmoid(torch.matmul(graph, self.i_z(x)))  # (..., N, D)
                m = torch.matmul(graph, self.i_m(x))  # (..., N, D)
                m = torch.tanh(m + r * self.h_m(x_shifted))
                m = self.dropout(m)
                x = (1 - z) * m + z * x

        if self.activation == 'relu':
            x = torch.relu(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'leaky':
            x = F.leaky_relu(x)

        return x