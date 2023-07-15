import torch
import torch.nn as nn
from models.utils import Segmentation


class TCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.window = args.window
        self.hidden = args.hidden
        self.kernel = args.kernels
        self.dilation = args.dilation
        self.layers = args.layers
        self.seg = args.seg
        self.channels = args.n_channels
        self.preprocess = args.preprocess

        if self.preprocess == 'seg':
            self.segmentation = Segmentation(self.seg, self.hidden, self.channels)
            self.dim = self.hidden
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2

        self.layer = nn.ModuleList()
        self.merge0 = nn.Conv2d(self.dim, self.hidden, kernel_size=(1, self.window // self.seg))
        self.merge = nn.ModuleList()
        for l in range(self.layers):
            conv = nn.Conv2d(self.hidden if l > 0 else self.dim, self.hidden, kernel_size=(1, self.kernel),
                             dilation=self.dilation)
            merge = nn.Conv2d(self.hidden, self.hidden,
                              kernel_size=(1, self.window // self.seg - (l + 1) * ((self.kernel - 1) * self.dilation)))
            self.layer.append(conv)
            self.merge.append(merge)

        self.decoder = nn.Sequential(nn.Linear(self.channels * (self.layers + 1) * self.hidden, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

    def encoder(self, x):
        z = [self.merge0(x)]
        for l in range(self.layers):
            x = self.layer[l](x)  # (B, D, C, T)
            h = self.merge[l](x)  # (B, D, C, 1)
            z.append(h)
        z = torch.cat(z, dim=-1)  # (B, D, C, L)
        return z

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.permute(0, 3, 2, 1)  # (B, D, C, T)
        z = self.encoder(x)

        z = z.reshape(bs, -1)  # (B, D*C*L)
        z = torch.relu(z)
        z = self.decoder(z).squeeze()
        return z
