import torch
import torch.nn as nn
from models.utils import Segmentation


class TCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.window = args.window * args.sample_rate
        self.hidden = args.hidden
        self.kernel = args.kernels
        self.dilation = args.dilation
        self.layers = args.layers
        self.seg = args.seg
        self.channels = args.n_channels

        if self.seg > 0:
            self.kernel = 5
            self.dilation = 1
            self.segmentation = Segmentation(self.seg, self.hidden, self.channels)
            seq_len = self.window // self.seg
            self.dim = self.hidden
        else:
            seq_len = self.window
            self.dim = 1

        self.layer = nn.ModuleList()
        self.merge0 = nn.Conv1d(self.dim, self.hidden, kernel_size=seq_len)
        self.merge = nn.ModuleList()
        for l in range(self.layers):
            conv = nn.Conv1d(self.hidden if l > 0 else self.dim, self.hidden, kernel_size=self.kernel, dilation=self.dilation)
            merge = nn.Conv1d(self.hidden, self.hidden, kernel_size=seq_len - (l + 1) * ((self.kernel - 1) * self.dilation))
            self.layer.append(conv)
            self.merge.append(merge)

        self.decoder = nn.Sequential(nn.Linear(self.channels * (self.layers + 1) * self.hidden, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

    def encoder(self, x):
        z = [self.merge0(x)]
        for l in range(self.layers):
            x = self.layer[l](x)  # (B, C, T'')
            h = self.merge[l](x)  # (B, C, 1)
            z.append(h)
        z = torch.cat(z, dim=-1)  # (B, C, L)
        return z

    def forward(self, x):
        u, x = x  # (B), (B, T, C)
        bs = x.shape[0]

        if self.seg > 0:
            x = self.segmentation.segment(x)  # (B, T', C, D)
            x = x.permute(0, 2, 1, 3).reshape(bs * self.channels, -1, self.hidden)  # (B*C, T', D)
        else:
            x = x.permute(0, 2, 1).reshape(bs * self.channels, -1).unsqueeze(dim=-1)  # (B*C, T', 1)

        x = x.permute(0, 2, 1)  # (B*C, D, T')
        z = self.encoder(x)  # (B*C, D, L)

        z = z.reshape(bs, -1)  # (B, C*D*L)
        z = self.decoder(z).squeeze()
        return z
