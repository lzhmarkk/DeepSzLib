import torch
import torch.nn as nn
from models.utils import Segmentation


class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.hidden = args.hidden
        self.window = args.window
        self.layers = args.layers
        self.channels = args.n_channels
        self.cell = args.cell
        self.preprocess = args.preprocess

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2

        if self.cell == 'RNN':
            self.encoder = nn.RNN(self.dim, self.hidden, self.layers)
        elif self.cell == 'LSTM':
            self.encoder = nn.LSTM(self.dim, self.hidden, self.layers)
        elif self.cell == 'GRU':
            self.encoder = nn.GRU(self.dim, self.hidden, self.layers)
        else:
            raise ValueError()

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

    def forward(self, x):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, S) -> (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.permute(1, 0, 2, 3).reshape(self.window // self.seg, bs * self.channels, self.dim)  # (T, B*C, D)
        z, h = self.encoder(x)
        z = z[-1, :, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)

        z = torch.relu(z)
        z = self.decoder(z).squeeze()  # (B)
        return z
