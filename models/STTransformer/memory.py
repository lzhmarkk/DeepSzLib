import torch
import numpy as np
import torch.nn as nn


class TimeEncode(nn.Module):
    def __init__(self, dim, seq_len):
        super(TimeEncode, self).__init__()

        self.seq_len = seq_len
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim))).float())
        self.phase = nn.Parameter(torch.zeros(dim).float())

    def forward(self, device):
        ts = torch.arange(self.seq_len).to(device).unsqueeze(-1)
        map_ts = ts * self.basis_freq.view(1, -1)  # [L, time_dim]
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class MemoryNetwork(nn.Module):
    def __init__(self, dim, num_nodes, seq_len, init_func, msg_func, upd_func, activation, dropout):
        super().__init__()

        self.dim = dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.init_func = init_func
        self.msg_func = msg_func
        self.upd_func = upd_func
        self.activation = activation
        self.dropout = dropout

        # initialization
        if self.init_func == 'train':
            self.node_init_embedding = nn.Parameter(torch.randn(self.num_nodes, self.dim))
        elif self.init_func == 'fc':
            self.node_init_weight = nn.Parameter(torch.randn(self.seq_len, self.num_nodes))
            self.node_init_fc = nn.Linear(self.dim, self.dim)
        elif self.init_func == 'attn':
            self.node_init_query = nn.Parameter(torch.randn(self.num_nodes, self.dim))
            self.node_init_fc = nn.Linear(self.dim, self.dim)

        # message
        if self.msg_func == 'id':
            self.msg_dim = self.dim
        elif self.msg_func == 'cat':
            self.msg_dim = 3 * self.dim
        elif self.msg_func == 'mlp':
            self.msg_mlp = nn.Sequential(nn.Linear(3 * self.dim, self.dim),
                                         nn.GELU())
            self.msg_dim = self.dim
        self.msg_dropout = nn.Dropout(self.dropout)

        # update
        if self.upd_func == 'gate':
            self.gamma = 0.8
            self.update_fc = nn.Linear(self.msg_dim, self.dim)
        elif self.upd_func == 'gru':
            self.gru_cell = nn.GRUCell(self.msg_dim, self.dim)

        # time encoding
        self.time_enc = TimeEncode(self.dim, self.seq_len)

    def get_initial_states(self, x):
        # (T, B, C, D)
        bs = x.shape[1]

        if self.init_func == 'zero':
            initial_node_state = torch.zeros([bs, self.num_nodes, self.dim]).to(x.device)
        elif self.init_func == 'random':
            initial_node_state = torch.randn([bs, self.num_nodes, self.dim]).to(x.device)
        elif self.init_func == 'train':
            initial_node_state = self.node_init_embedding.data.unsqueeze(dim=0).expand(bs, -1, -1)
        elif self.init_func == 'mean':
            initial_node_state = torch.mean(x, dim=0)
        elif self.init_func == 'fc':
            initial_node_state = torch.einsum("tbcd, tc->bcd", x, self.node_init_weight)
            initial_node_state = self.node_init_fc(initial_node_state)
        elif self.init_func == 'attn':
            weight = torch.einsum('cd,tbcd->tbc', self.node_init_query, x) / np.sqrt(self.dim)
            weight = torch.softmax(weight, dim=0)  # (T, B, C)
            initial_node_state = torch.einsum('tbc,tbcd->bcd', weight, x)
            initial_node_state = self.node_init_fc(initial_node_state)
        else:
            raise ValueError(f"Not supported initialization method {self.init_func}")

        return initial_node_state

    def get_message(self, x, s, t):
        # (B, C, D)
        t = t.unsqueeze(0).repeat(x.shape[0], 1)

        if self.msg_func == 'id':
            msg = x
        elif self.msg_func == 'cat':
            msg = torch.cat([x, s, t], dim=-1)
        elif self.msg_func == 'mlp':
            msg = self.msg_mlp(torch.cat([x, s, t], dim=-1))
        else:
            raise ValueError(f"Not supported message {self.init_func}")

        msg = self.msg_dropout(msg)
        return msg

    def update_node_state(self, s, msg):
        # (B, C, D), (B, C, D)
        if self.upd_func == 'gate':
            s = (1 - self.gamma) * s + self.gamma * self.update_fc(msg)
        elif self.upd_func == 'gru':
            s = self.gru_cell(msg, s)
        return s

    def forward(self, x, mask=None):
        # (T, B, C, D), (T)
        if mask is None:
            mask = torch.ones(x.shape[0]).int().to(x.device)

        # initialize node states
        node_state = self.get_initial_states(x).reshape(-1, self.dim)  # (B*C, D)

        # time encoding
        t_enc = self.time_enc(x.device)  # (T)

        # update node states
        node_state_all = []
        time_range = torch.arange(self.seq_len).to(x.device) * mask.to(x.device)
        for t in time_range:
            message = self.get_message(x[t].reshape(-1, self.dim), node_state, t_enc[t])  # (B*C, D)
            node_state = self.update_node_state(node_state, message)
            node_state_all.append(node_state)

        node_state_all = torch.stack(node_state_all, dim=0).reshape(mask.shape[0], *x.shape[1:])
        return node_state_all  # (T, B, C, D)
