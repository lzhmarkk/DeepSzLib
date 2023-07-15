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
        self.multi_task = args.multi_task

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
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        if self.multi_task:
            self.horizon = args.horizon
            if self.cell == 'RNN':
                self.predictor = nn.ModuleList([nn.RNNCell(self.hidden, self.hidden) for _ in range(self.layers)])
            elif self.cell == 'LSTM':
                self.predictor = nn.ModuleList([nn.LSTMCell(self.hidden, self.hidden) for _ in range(self.layers)])
            elif self.cell == 'GRU':
                self.predictor = nn.ModuleList([nn.GRUCell(self.hidden, self.hidden) for _ in range(self.layers)])
            else:
                raise ValueError()
            self.fc = nn.Linear(self.hidden, self.dim)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, S) -> (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.permute(1, 0, 2, 3).reshape(self.window // self.seg, bs * self.channels, self.dim)  # (T, B*C, D)
        z, h = self.encoder(x)  # (T, B*C, D), (L, B*C, D)

        # decoder
        z = z.mean(dim=0)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)
        z = torch.tanh(z)
        z = self.decoder(z).squeeze()  # (B)

        if not self.multi_task:
            return z
        else:
            output = []
            y = torch.zeros_like(h[0])  # go_symbol, (B*C, D)
            hidden_states = [h[_] for _ in range(self.layers)]  # copy states
            for t in range(self.horizon//self.seg):
                for l in range(self.layers):
                    if self.cell == 'LSTM':
                        y, hidden = self.predictor[l](y, hidden_states[l])  # (B*C, D)
                    else:
                        y = self.predictor[l](y, hidden_states[l])  # (B*C, D)
                        hidden = y
                    hidden_states[l] = hidden
                output.append(y)
            y = torch.stack(output, dim=0)  # (T, B*C, D)
            y = self.fc(y)
            y = y.reshape(self.horizon//self.seg, bs, self.channels, self.dim).permute(1,0,2,3)
            return z, y
