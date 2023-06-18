import torch
import torch.nn as nn
from models.utils import Segmentation


class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.hidden = args.hidden
        self.layers = args.layers
        self.channels = args.n_channels
        self.cell = args.cell

        if self.seg > 0:
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        else:
            self.dim = 1

        if self.cell == 'RNN':
            self.encoder = nn.RNN(self.dim, self.hidden, self.layers, batch_first=True)
        elif self.cell == 'LSTM':
            self.encoder = nn.LSTM(self.dim, self.hidden, self.layers, batch_first=True)
        elif self.cell == 'GRU':
            self.encoder = nn.GRU(self.dim, self.hidden, self.layers, batch_first=True)
        else:
            raise ValueError()

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

    def forward(self, x):
        # (B), (B, T, C)
        u, x = x
        bs = x.shape[0]

        if self.seg > 0:
            x = self.segmentation.segment(x)  # (B, T', C, D)
        else:
            x = x.unsqueeze(dim=-1)  # (B, T', C, 1)

        x = x.permute(0, 2, 1, 3).reshape(bs * self.channels, -1, self.dim)  # (B*C, T', D)
        z, h = self.encoder(x)
        z = z[:, -1, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)

        z = self.decoder(z)  # (B)
        z = z.squeeze(-1)
        # z = torch.sigmoid(z.squeeze(-1))
        return z
