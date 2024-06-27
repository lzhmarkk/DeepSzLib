import torch
import torch.nn as nn
from models.utils import Patching, check_tasks


class TSD(nn.Module):
    supported_tasks = ['detection', 'onset_detection', 'classification', 'prediction']
    unsupported_tasks = []

    def __init__(self, args):
        super().__init__()
        self.patch_len = args.patch_len
        self.window = args.window
        self.horizon = args.horizon
        self.hidden = args.hidden
        self.seq_len = self.window // self.patch_len
        self.layers = args.layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.dropout = args.dropout
        self.position_encoding = args.pos_enc
        self.preprocess = args.preprocess
        self.task = args.task
        self.onset_history_len = args.onset_history_len
        check_tasks(self)

        if self.preprocess == 'raw':
            self.dim = self.hidden
            self.patching = Patching(self.patch_len, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.patch_len // 2
            self.fc = nn.Linear(self.channels * self.dim, self.hidden)

        if self.position_encoding:
            self.pos_emb = nn.Parameter(torch.randn([self.seq_len + 1, self.hidden]), requires_grad=True)

        transformer_layer = nn.TransformerEncoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
        self.encoder = nn.TransformerEncoder(transformer_layer, self.layers)

        self.decoder = nn.Linear(self.hidden, args.n_classes)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden), requires_grad=True)

        if 'prediction' in self.task:
            self.pred_pos_emb = nn.Parameter(torch.randn([self.horizon // self.patch_len, self.hidden]), requires_grad=True)

            transformer_decoder_layer = nn.TransformerDecoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
            self.pred_decoder = nn.TransformerDecoder(transformer_decoder_layer, self.layers)
            self.pred_fc = nn.Linear(self.hidden, self.channels * self.dim)

    def predict(self, x):
        bs = x.shape[0]
        x = torch.cat([x, self.cls_token.expand(bs, -1, -1)], dim=1)

        if self.position_encoding:
            x = x + self.pos_emb[-x.shape[1]:].unsqueeze(0)  # (B, T+1, D)

        x = x.permute(1, 0, 2).reshape(-1, bs, self.hidden)  # (T+1, B, D)
        mask = torch.triu(torch.ones(x.shape[0], x.shape[0], device=x.device) * float('-inf'), diagonal=1)
        h = self.encoder(x, mask=mask)  # (T+1, B, D)
        z = h[-1, :, :]  # (B, D)
        z = z.reshape(bs, self.hidden)  # (B, D)
        z = self.decoder(z).squeeze(dim=-1)  # (B)
        return z, h

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]
        if self.preprocess == 'raw':
            x = self.patching.patching(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x.reshape(bs, self.seq_len, self.channels * self.dim))  # (B, T, D)

        if 'onset_detection' in self.task:
            out = []
            for t in range(1, self.seq_len + 1):
                xt = x[:, max(0, t - self.onset_history_len):t, :]
                zt, h = self.predict(xt)
                out.append(zt)
            z = torch.stack(out, dim=1)

        elif 'detection' in self.task or 'classification' in self.task:
            z, h = self.predict(x)

        else:
            raise NotImplementedError

        if 'prediction' not in self.task:
            return {'prob': z}
        else:
            m = h[:-1, :, :]  # (T, B, D)
            y = self.pred_pos_emb.unsqueeze(dim=1).repeat(1, bs, 1).reshape(self.horizon // self.patch_len, bs, self.hidden)
            y = self.pred_decoder(y, m)
            y = y.reshape(self.horizon // self.patch_len, bs, self.hidden)  # (T, B, D)
            y = self.pred_fc(y)
            y = y.reshape(self.horizon // self.patch_len, bs, self.channels, self.dim)
            y = y.transpose(0, 1)  # (B, T, C, D)
            return {'prob': z, 'pred': y}
