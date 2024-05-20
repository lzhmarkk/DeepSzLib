import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Segmentation
from models.MTGNN.modules import mixprop, dilated_inception


class graph_constructor(nn.Module):
    def __init__(self, nnodes, dim, alpha=3):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

        self.dim = dim
        self.alpha = alpha

    def forward(self, x):
        nodevec1 = self.emb1.weight
        nodevec2 = self.emb2.weight

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        # mask = torch.zeros(self.nnodes, self.nnodes).to(x.device)
        # mask.fill_(float('0'))
        # s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        # mask.scatter_(1, t1, s1.fill_(1))
        # adj = adj * mask
        return adj


class MTGNN(nn.Module):
    def __init__(self, args):
        super(MTGNN, self).__init__()
        self.num_nodes = args.n_channels
        self.gcn_depth = args.gcn_depth
        self.dropout = args.dropout
        self.window = args.window
        self.seq_length = args.window // args.seg
        self.layers = args.layers
        self.hidden = args.hidden
        self.seg = args.seg
        self.preprocess = args.preprocess

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.task = args.task
        self.anomaly_len = args.anomaly_len
        if 'detection' not in self.task:
            self.seq_length = self.anomaly_len

        if self.preprocess == 'raw':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        self.start_conv = nn.Conv2d(self.dim, self.hidden, kernel_size=(1, 1))
        self.gc = graph_constructor(self.num_nodes, args.node_dim, args.tanhalpha)

        kernel_size = 7
        self.receptive_field = self.layers * (kernel_size - 1) + 1

        for j in range(1, self.layers + 1):
            rf_size_j = 1 + j * (kernel_size - 1)
            self.filter_convs.append(dilated_inception(self.hidden, self.hidden, dilation_factor=1))
            self.gate_convs.append(dilated_inception(self.hidden, self.hidden, dilation_factor=1))
            self.residual_convs.append(nn.Conv2d(self.hidden, self.hidden, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(self.hidden, self.hidden, kernel_size=(1, max(self.seq_length, self.receptive_field) - rf_size_j + 1)))
            self.gconv1.append(mixprop(self.hidden, self.hidden, self.gcn_depth, self.dropout, args.propalpha))
            self.gconv2.append(mixprop(self.hidden, self.hidden, self.gcn_depth, self.dropout, args.propalpha))
            self.norm.append(nn.LayerNorm([self.hidden, self.num_nodes, max(self.seq_length, self.receptive_field) - rf_size_j + 1]))

        self.skip0 = nn.Conv2d(self.dim, self.hidden, kernel_size=(1, max(self.seq_length, self.receptive_field)))
        self.skipE = nn.Conv2d(self.hidden, self.hidden, kernel_size=(1, max(1, self.seq_length - self.receptive_field + 1)))

        assert 'prediction' not in self.task
        assert 'detection' in self.task or 'onset_detection' in self.task

        self.decoder = nn.Sequential(nn.Linear(self.num_nodes * self.hidden, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

    def predict(self, x):
        bs = x.shape[0]
        adp = self.gc(x)

        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv(x)
        for i in range(self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            skip = skip + self.skip_convs[i](x)

            x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x)

        skip = self.skipE(x) + skip
        z = F.relu(skip)

        z = z.reshape(bs, -1)
        z = self.decoder(z).squeeze(-1)
        return z

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        if self.preprocess == 'raw':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.transpose(3, 1)  # (bs, input_dim, num_nodes, window)

        if 'detection' in self.task:
            if self.seq_length < self.receptive_field:
                x = nn.functional.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))

            z = self.predict(x)

        else:
            out = []
            for t in range(1, self.window // self.seg + 1):
                xt = x[:, :, :, max(0, t - self.anomaly_len):t]
                if xt.shape[-1] < self.receptive_field:
                    xt = nn.functional.pad(xt, (self.receptive_field - xt.shape[-1], 0, 0, 0))

                z = self.predict(xt)
                out.append(z)
            z = torch.stack(out, dim=1)

        return z, None
