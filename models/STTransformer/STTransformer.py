import torch
import torch.nn as nn
from models.utils import Segmentation
from .Encoder import SpatialTemporalEncoder
from models.DCRNN.graph import distance_support, correlation_support, norm_graph


class STTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.window = args.window
        self.hidden = args.hidden
        self.layers = args.layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.dropout = args.dropout
        self.position_encoding = args.pos_enc
        self.preprocess = args.preprocess
        self.use_support = args.use_support
        self.filter_type = args.filter_type
        self.multi_task = args.multi_task

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        if self.position_encoding:
            self.pos_emb = nn.Parameter(torch.randn([1 + self.window // self.seg, self.channels, self.hidden]), requires_grad=True)

        self.encoder = SpatialTemporalEncoder(layers=self.layers, hidden=self.hidden, heads=self.heads, dropout=self.dropout,
                                              seq_len=1 + self.window // self.seg, n_channels=self.channels, filter_type=self.filter_type)

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.channels, self.hidden), requires_grad=True)

    def get_support(self, x):
        if self.use_support == 'dist':
            if not hasattr(self, 'supports'):
                support = distance_support(self.channels)
                support = norm_graph(support, self.filter_type)
                support = [s.to(x.device) for s in support]
                self.supports = support
            supports = self.supports
        elif self.use_support == 'corr':
            supports = []
            for _x in x:
                support = correlation_support(_x.cpu().numpy())
                support = norm_graph(support, self.filter_type)
                support = torch.stack([s.to(x.device) for s in support], dim=0)
                supports.append(support)
            supports = torch.stack(supports, dim=0).transpose(1, 0)

        else:
            raise ValueError()

        return supports

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        graphs = self.get_support(x)

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x)  # (B, T, C, D)

        x = torch.cat([self.cls_token.expand(bs, -1, -1, -1), x], dim=1)

        if self.position_encoding:
            x = x + self.pos_emb.unsqueeze(0)  # (B, 1+T, C, D)

        x = x.permute(1, 0, 2, 3).reshape(1 + self.window // self.seg, bs * self.channels, self.hidden)  # (1+T, B*C, D)
        z = self.encoder(x, graphs)  # (1+T, B*C, D)

        # decoder
        z = z[0, :, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)

        z = torch.tanh(z)
        z = self.decoder(z).squeeze()  # (B)

        return z
