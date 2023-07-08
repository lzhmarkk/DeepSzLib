import torch
import torch.nn as nn
from models.utils import Segmentation


class RNNTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.hidden = args.hidden
        self.window = args.window
        self.rnn_layers = args.rnn_layers
        self.attn_layers = args.attn_layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.cell = args.cell
        self.dropout = args.dropout
        self.preprocess = args.preprocess

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2

        if self.cell == 'RNN':
            self.rnn = nn.RNN(self.dim, self.hidden, self.rnn_layers)
        elif self.cell == 'LSTM':
            self.rnn = nn.LSTM(self.dim, self.hidden, self.rnn_layers)
        elif self.cell == 'GRU':
            self.rnn = nn.GRU(self.dim, self.hidden, self.rnn_layers)
        else:
            raise ValueError()

        transformer_layer = nn.TransformerEncoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
        self.attn = nn.TransformerEncoder(transformer_layer, self.attn_layers)

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.channels, self.hidden), requires_grad=True)

    def forward(self, x):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, S) -> (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        # RNN
        x = x.permute(1, 0, 2, 3).reshape(self.window // self.seg, bs * self.channels, self.dim)  # (T, B*C, D)
        z, h = self.rnn(x)

        # Attention
        z = z.reshape(self.window // self.seg, bs, self.channels, self.hidden)  # (T, B, C, D)
        z = torch.cat([self.cls_token.expand(-1, bs, -1, -1), z], dim=0)  # (1+T, B, C, D)
        z = z.reshape(1 + self.window // self.seg, bs * self.channels, self.hidden)  # (1+T, B*C, D)
        z = self.attn(z)  # (1+T, B*C, D)
        z = z[0, :, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)

        # decoder
        z = torch.tanh(z)
        z = self.decoder(z).squeeze()  # (B)
        return z
