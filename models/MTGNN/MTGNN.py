import torch
import torch.nn as nn
import torch.nn.functional as fn
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
        adj = fn.relu(torch.tanh(self.alpha * a))
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

        if self.preprocess == 'seg':
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

        self.decoder = nn.Sequential(nn.Linear(self.num_nodes * self.hidden, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

    def forward(self, x):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.transpose(3, 1)  # (bs, input_dim, num_nodes, window)
        if self.seq_length < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))

        adp = self.gc(x)

        skip = self.skip0(fn.dropout(x, self.dropout, training=self.training))
        x = self.start_conv(x)
        for i in range(self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate
            x = fn.dropout(x, self.dropout, training=self.training)
            skip = skip + self.skip_convs[i](x)

            x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x)

        skip = self.skipE(x) + skip
        z = fn.relu(skip)

        z = z.reshape(bs, -1)
        z = self.decoder(z).squeeze()
        return z
