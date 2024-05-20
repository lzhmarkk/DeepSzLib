import torch
import torch.nn as nn
from models.utils import Patching
from models.utils import check_tasks


class SegRNN(nn.Module):
    supported_tasks = ['detection', 'onset_detection', 'classification', 'prediction']
    unsupported_tasks = []

    def __init__(self, args):
        super().__init__()
        self.patch_len = args.patch_len
        self.hidden = args.hidden
        self.window = args.window
        self.layers = args.layers
        self.channels = args.n_channels
        self.cell = args.cell
        self.preprocess = args.preprocess
        self.task = args.task
        self.task = args.task
        check_tasks(self)

        if self.preprocess == 'raw':
            self.dim = self.hidden
            self.patching = Patching(self.patch_len, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.patch_len // 2

        if self.cell == 'RNN':
            self.encoder = nn.RNN(self.channels * self.dim, self.hidden, self.layers)
        elif self.cell == 'LSTM':
            self.encoder = nn.LSTM(self.channels * self.dim, self.hidden, self.layers)
        elif self.cell == 'GRU':
            self.encoder = nn.GRU(self.channels * self.dim, self.hidden, self.layers)
        else:
            raise ValueError()

        self.decoder = nn.Linear(self.hidden, args.n_classes)

        if 'prediction' in self.task:
            self.horizon = args.horizon
            if self.cell == 'RNN':
                self.predictor = nn.ModuleList([nn.RNNCell(self.hidden, self.hidden) for _ in range(self.layers)])
            elif self.cell == 'LSTM':
                self.predictor = nn.ModuleList([nn.LSTMCell(self.hidden, self.hidden) for _ in range(self.layers)])
            elif self.cell == 'GRU':
                self.predictor = nn.ModuleList([nn.GRUCell(self.hidden, self.hidden) for _ in range(self.layers)])
            else:
                raise ValueError()

            self.fc = nn.Linear(self.hidden, self.channels * self.dim)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'raw':
            x = self.patching.patching(x)  # (B, T, C, S) -> (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.permute(1, 0, 2, 3).reshape(self.window // self.patch_len, bs, self.channels * self.dim)  # (T, B, C*D)
        z, h = self.encoder(x)  # (T, B, D), (L, B, D)

        # decoder
        if 'onset_detection' in self.task:
            z = z.transpose(0, 1)
            z = z.reshape(bs, self.window // self.patch_len, self.hidden)  # (B, T, D)
            z = self.decoder(z).squeeze(dim=-1)  # (B, T)
        elif 'detection' in self.task or 'classification' in self.task:
            z = z[-1, :, :]
            z = z.reshape(bs, self.hidden)  # (B, D)
            z = self.decoder(z).squeeze(dim=-1)  # (B)
        else:
            raise NotImplementedError

        if 'prediction' not in self.task:
            return z, None
        else:
            output = []
            y = torch.zeros_like(h[0])  # go_symbol, (B, D)
            hidden_states = [h[_] for _ in range(self.layers)]  # copy states
            for t in range(self.horizon // self.patch_len):
                for l in range(self.layers):
                    if self.cell == 'LSTM':
                        y, hidden = self.predictor[l](y, hidden_states[l])  # (B, D)
                    else:
                        y = self.predictor[l](y, hidden_states[l])  # (B, D)
                        hidden = y
                    hidden_states[l] = hidden
                output.append(y)
            y = torch.stack(output, dim=0)  # (T, B, D)
            y = self.fc(y)
            y = y.reshape(self.horizon // self.patch_len, bs, self.channels, self.dim).permute(1, 0, 2, 3)
            return z, y
