import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Patching
from models.DCRNN.graph import distance_support, norm_graph
from models.utils import check_tasks


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    """
    supported_tasks = ['detection', 'onset_detection', 'classification']
    unsupported_tasks = ['prediction']

    def __init__(self, args):
        super(STGCN, self).__init__()
        self.window = args.window
        self.num_nodes = args.n_channels
        self.channels = args.channels
        self.hidden = args.hidden
        self.spatial_channels = args.spatial_channels
        self.preprocess = args.preprocess
        self.patch_len = args.patch_len
        self.filter_type = args.filter_type
        self.onset_history_len = args.onset_history_len
        self.task = args.task
        check_tasks(self)

        if self.preprocess == 'raw':
            self.dim = self.hidden
            self.patching = Patching(self.patch_len, self.hidden, self.num_nodes)
        elif self.preprocess == 'fft':
            self.dim = self.patch_len // 2

        self.receptive_field = 11
        self.block1 = STGCNBlock(in_channels=self.dim, out_channels=self.hidden,
                                 spatial_channels=self.spatial_channels, num_nodes=self.num_nodes)
        self.block2 = STGCNBlock(in_channels=self.hidden, out_channels=self.hidden,
                                 spatial_channels=self.spatial_channels, num_nodes=self.num_nodes)
        self.last_temporal = TimeBlock(in_channels=self.hidden, out_channels=self.hidden)

        self.decoder = nn.Sequential(nn.Linear(self.num_nodes * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, args.n_classes))

    def get_support(self, x):
        if not hasattr(self, 'support'):
            support = distance_support(self.channels)
            support = norm_graph(support, self.filter_type)
            support = [s.to(x.device) for s in support]
            self.support = support[0]

        return self.support

    def predict(self, x, adj_mx):
        bs = x.shape[0]
        x = self.block1(x.permute(0, 2, 1, 3), adj_mx)
        x = self.block2(x, adj_mx)
        z = self.last_temporal(x)

        z = torch.mean(z, dim=2)
        z = self.decoder(z.reshape(bs, -1)).squeeze(dim=-1)  # (B)
        return z

    def forward(self, x, p, y):
        # (B, T, C, D)
        adj_mx = self.get_support(x)

        if 'detection' in self.task or 'classification' in self.task:
            z = self.predict(x, adj_mx)

        elif 'onset_detection' in self.task:
            out = []
            for t in range(1, self.window // self.patch_len + 1):
                xt = x[:, max(0, t - self.onset_history_len):t, :, :]
                if xt.shape[1] < self.receptive_field:
                    xt = nn.functional.pad(xt, (0, 0, 0, 0, self.receptive_field - xt.shape[1], 0))

                z = self.predict(xt, adj_mx)
                out.append(z)
            z = torch.stack(out, dim=1)

        else:
            raise NotImplementedError

        return {'prob': z}
