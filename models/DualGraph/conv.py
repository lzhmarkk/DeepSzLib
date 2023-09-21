import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGNN(nn.Module):
    def __init__(self, dim, n_nodes, n_layers, dropout, method, activation, depth=1, local_gnn_decay=1):
        super().__init__()

        self.dim = dim
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.depth = depth
        self.activation = activation
        self.local_gnn_decay = local_gnn_decay

        self.method = method
        if self.method == 'gcn':
            self.w = nn.Linear((self.depth + 1) * self.dim, self.dim)

        elif self.method == 'sage':
            self.w = nn.ModuleList()
            self.ln = nn.ModuleList()
            for l in range(self.depth):
                self.w.append(nn.Linear(2 * self.dim, self.dim))
                if l < self.depth - 1:
                    self.ln.append(nn.LayerNorm(self.dim))

        elif self.method == 'gat':
            self.a1 = nn.ModuleList()
            self.a2 = nn.ModuleList()
            self.w = nn.ModuleList()
            self.ln = nn.ModuleList()
            for l in range(self.depth):
                self.a1.append(nn.Linear(self.dim, 1))
                self.a2.append(nn.Linear(self.dim, 1))
                self.w.append(nn.Linear(self.dim, self.dim))
                if l < self.depth - 1:
                    self.ln.append(nn.LayerNorm(self.dim))

        elif self.method == 'gnn':
            self.i_r = nn.Linear(self.dim, self.dim)
            self.h_r = nn.Linear(self.dim, self.dim)
            self.i_z = nn.Linear(self.dim, self.dim)
            self.h_z = nn.Linear(self.dim, self.dim)
            self.i_m = nn.Linear(self.dim, self.dim)
            self.h_m = nn.Linear(self.dim, self.dim)

        elif self.method == 'rnn':
            self.h_r = nn.Linear(self.dim, self.dim)
            self.x_r = nn.Linear(self.dim, self.dim)
            self.h_z = nn.Linear(self.dim, self.dim)
            self.x_z = nn.Linear(self.dim, self.dim)
            self.h_m = nn.Linear(self.dim, self.dim)
            self.x_m = nn.Linear(self.dim, self.dim)

        elif self.method == 'grnn':
            self.i_r = nn.Linear(self.dim, self.dim)
            self.h_r = nn.Linear(self.dim, self.dim)
            self.x_r = nn.Linear(self.dim, self.dim)
            self.i_z = nn.Linear(self.dim, self.dim)
            self.h_z = nn.Linear(self.dim, self.dim)
            self.x_z = nn.Linear(self.dim, self.dim)
            self.i_m = nn.Linear(self.dim, self.dim)
            self.h_m = nn.Linear(self.dim, self.dim)
            self.x_m = nn.Linear(self.dim, self.dim)

        elif self.method == 'evo':
            self.gamma = 1 / self.local_gnn_decay
            self.q = nn.Linear(self.dim, self.dim)
            self.k = nn.Linear(self.dim, self.dim)
            self.v = nn.Linear(self.dim, self.dim)

            self.w0 = nn.Parameter(torch.randn(1, self.dim))
            self.a0 = nn.Parameter(torch.randn(1, self.dim))

            self.h_r = nn.Linear(self.dim, self.dim)
            self.x_r = nn.Linear(self.dim, self.dim)
            self.h_z = nn.Linear(self.dim, self.dim)
            self.x_z = nn.Linear(self.dim, self.dim)
            self.h_m = nn.Linear(self.dim, self.dim)
            self.x_m = nn.Linear(self.dim, self.dim)

            self.out_proj = nn.Linear(self.dim, self.dim)

        elif self.method == 'mm':
            self.w = nn.Linear(self.dim, self.dim)

        self.dropout = nn.Dropout(dropout)
        self.eye = torch.eye(n_nodes).bool()

        self.reset_parameters()

        self.norm = nn.LayerNorm(self.dim)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)

        if self.method == 'rnn':
            for module in [self.h_r, self.x_r, self.h_z, self.x_z, self.h_m, self.x_m]:
                for weight in module.parameters():
                    nn.init.uniform_(weight, -stdv, stdv)

        elif self.method == 'grnn':
            for module in [self.h_r, self.x_r, self.h_z, self.x_z, self.h_m, self.x_m]:
                for weight in module.parameters():
                    nn.init.uniform_(weight, -stdv, stdv)
            for module in [self.i_r, self.i_z, self.i_m]:
                for weight in module.parameters():
                    nn.init.uniform_(weight, -stdv / self.local_gnn_decay, stdv / self.local_gnn_decay)

        elif self.method == 'evo':
            for module in [self.h_r, self.x_r, self.h_z, self.x_z, self.h_m, self.x_m]:
                for weight in module.parameters():
                    nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, graph):
        # (..., N, D), (..., N, N)

        if self.method == 'gcn':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)
            graph = graph + eye
            D = 1 / (torch.sqrt(graph.sum(dim=-1)) + 1e-7)
            D = torch.diag_embed(D)  # (..., N, N)
            graph = torch.matmul(D, torch.matmul(graph, D))  # (..., N, N)
            graph = self.dropout(graph)

            out = [x]
            for l in range(self.depth):
                x = torch.matmul(graph, x)
                out.append(x)
            x = self.w(torch.cat(out, dim=-1))

        elif self.method == 'sage':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)
            graph = graph * (~eye)  # Drop self-connection
            graph = graph / (torch.sum(graph, dim=-1, keepdim=True) + 1e-7)  # (..., N, N)
            graph = self.dropout(graph)

            for l in range(self.depth):
                neighbor = torch.matmul(graph, x)  # (..., N, N)
                x = self.w[l](torch.cat([x, neighbor], dim=-1))
                if l < self.depth - 1:
                    x = self.ln[l](x)

        elif self.method == 'gat':
            for l in range(self.depth):
                adj = F.leaky_relu(self.a1[l](x) + self.a2[l](x).transpose(-2, -1))  # (..., N, N)
                adj = self.dropout(adj)
                adj = torch.softmax(adj, dim=-1)  # (..., N, N)
                x = x + self.w[l](torch.matmul(adj, x))  # (..., N, D)
                if l < self.depth - 1:
                    x = self.ln[l](x)

        elif self.method == 'gnn':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)
            x_shifted = F.pad(x, (0, 0, 1, 0))[..., :-1, :]  # (..., N, D)
            graph = graph * (~eye)  # Drop self-connection
            diag = graph * eye

            r = torch.sigmoid(torch.matmul(graph, self.i_r(x)) + torch.matmul(diag, self.h_r(x)))  # (..., N, D)
            z = torch.sigmoid(torch.matmul(graph, self.i_z(x)) + torch.matmul(diag, self.h_z(x)))  # (..., N, D)
            m = torch.matmul(graph, self.i_m(x))  # (..., N, D)
            m = torch.tanh(m + r * self.h_m(x_shifted))
            m = self.dropout(m)
            x = (1 - z) * m + z * x

        elif self.method == 'rnn':
            h = torch.zeros_like(x[..., 0, :])  # (..., T, D)
            output = []
            for t in range(self.n_nodes):
                xt = x[..., t, :]
                r = torch.sigmoid(self.x_r(xt) + self.h_r(h))
                z = torch.sigmoid(self.x_z(xt) + self.h_z(h))
                m = torch.tanh(self.x_m(xt) + r * self.h_m(h))
                h = (1 - z) * m + z * h
                output.append(h)
            x = torch.stack(output, dim=-2)  # (..., T, D)

        elif self.method == 'grnn':
            eye = self.eye.reshape((1,) * (graph.ndim - 2) + (*self.eye.shape,)).to(x.device)
            graph = graph * (~eye)

            h = torch.zeros_like(x[..., 0, :])  # (..., T, D)
            output = []
            for t in range(self.n_nodes):
                xt = x[..., t, :]
                g = graph[..., t, :].unsqueeze(dim=-2)  # (..., 1, N)
                assert (g[..., 1 + t:] == 0).all(), f"Information leaks!"
                assert (g[..., t] == 0).all(), f"Diagonal not masked!"
                r = torch.sigmoid(torch.matmul(g, self.i_r(x)).squeeze(dim=-2) + self.x_r(xt) + self.h_r(h))
                z = torch.sigmoid(torch.matmul(g, self.i_z(x)).squeeze(dim=-2) + self.x_z(xt) + self.h_z(h))
                m = torch.tanh(torch.matmul(g, self.i_m(x)).squeeze(dim=-2) + self.x_m(xt) + r * self.h_m(h))
                h = (1 - z) * m + z * h
                output.append(h)
            x = torch.stack(output, dim=-2)  # (..., T, D)

        elif self.method == 'evo':
            bs = x.shape[0]
            h = torch.zeros_like(x[..., 0, :])  # (..., T, D)
            s = torch.zeros(bs, self.dim, self.dim, device=x.device)
            m = torch.zeros(bs, 1, self.dim, device=x.device)

            output = []
            attn_weight, attn_value = [], []
            for t in range(self.n_nodes):
                xt = x[..., t, :]
                query = self.q(xt).unsqueeze(dim=-2)  # (B, 1, D)
                key = self.k(xt)  # (B, D, 1)
                value = self.v(xt).unsqueeze(dim=-2)  # (B, 1, D)

                fn = lambda x: torch.exp(x)  # This activation may lead to huge values, e.g. 1e26
                query = fn(query)
                key = fn(key)

                a = self.a0.sigmoid()
                w = self.w0.sigmoid()

                s = a * s + w * key.unsqueeze(dim=-1) * value  # (B, D, D)
                m = a * m + w * key.unsqueeze(dim=1)  # (B, 1, D)

                div = torch.matmul(query, self.dropout(m).transpose(1, 2)) + 1e-5
                out = torch.matmul(query, self.dropout(s)) / div
                if hasattr(self, 'attn_weight'):  # for visualization
                    attn_weight.append(div.squeeze().clone().detach())
                if hasattr(self, 'attn_value'):
                    attn_value.append(out.clone().detach().sum(dim=-1))
                out = torch.where(torch.isinf(out), 0, out)
                out = out.squeeze(dim=1)

                out = self.norm(out)
                out = self.gamma * out

                r = torch.sigmoid(out + self.x_r(xt) + self.h_r(h))
                z = torch.sigmoid(out + self.x_z(xt) + self.h_z(h))
                n = torch.tanh(out + self.x_m(xt) + r * self.h_m(h))
                h = (1 - z) * n + z * h
                h = self.dropout(h)
                output.append(h)

            output = self.out_proj(torch.stack(output, dim=-2))  # (..., T, D)
            x = x + output
            if hasattr(self, 'attn_weight'):
                self.attn_weight.append(torch.stack(attn_weight, dim=1))
            if hasattr(self, 'attn_value'):
                self.attn_value.append(torch.stack(attn_value, dim=1))
            if hasattr(self, 'out_value'):
                self.out_value.append(output.clone().detach().sum(dim=-1))

        elif self.method == 'mm':
            x = x + self.w(torch.matmul(graph, x))

        if self.activation == 'relu':
            x = torch.relu(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'leaky':
            x = F.leaky_relu(x)

        return x


class GlobalGNN(LocalGNN):
    def __init__(self, dim, n_nodes, n_layers, dropout, method, activation, depth):
        super().__init__(dim, n_nodes, n_layers, dropout, method, activation, depth)

        assert method != 'rnn'
