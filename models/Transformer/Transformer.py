import torch
import torch.nn as nn
from models.utils import Segmentation


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.window = args.window
        self.horizon = args.horizon
        self.hidden = args.hidden
        self.layers = args.layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.dropout = args.dropout
        self.position_encoding = args.pos_enc
        self.preprocess = args.preprocess
        self.task = args.task

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.channels * self.dim, self.hidden)

        if self.position_encoding:
            self.pos_emb = nn.Parameter(torch.randn([1 + self.window // self.seg, self.hidden]), requires_grad=True)

        transformer_layer = nn.TransformerEncoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
        self.encoder = nn.TransformerEncoder(transformer_layer, self.layers)

        self.decoder = nn.Linear(self.hidden, 1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden), requires_grad=True)

        if 'pred' in self.task:
            self.pred_pos_emb = nn.Parameter(torch.randn([self.horizon // self.seg, self.hidden]), requires_grad=True)

            transformer_decoder_layer = nn.TransformerDecoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
            self.pred_decoder = nn.TransformerDecoder(transformer_decoder_layer, self.layers)
            self.pred_fc = nn.Linear(self.hidden, self.channels * self.dim)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x.reshape(bs, self.window // self.seg, self.channels * self.dim))  # (B, T, D)

        x = torch.cat([self.cls_token.expand(bs, -1, -1), x], dim=1)

        if self.position_encoding:
            x = x + self.pos_emb.unsqueeze(0)  # (B, 1+T, D)

        x = x.permute(1, 0, 2).reshape(1 + self.window // self.seg, bs, self.hidden)  # (1+T, B, D)
        h = self.encoder(x)  # (1+T, B, D)

        # decoder
        z = h[0, :, :]  # (B, D)
        z = z.reshape(bs, self.hidden)  # (B, D)

        # z = torch.tanh(z)
        z = self.decoder(z).squeeze(dim=-1)  # (B)

        if 'pred' not in self.task:
            return z, None
        else:
            m = h[1:, :, :]  # (T, B, D)
            y = self.pred_pos_emb.unsqueeze(dim=1).repeat(1, bs, 1).reshape(self.horizon // self.seg, bs, self.hidden)
            y = self.pred_decoder(y, m)
            y = y.reshape(self.horizon // self.seg, bs, self.hidden)  # (T, B, D)
            y = self.pred_fc(y)
            y = y.permute(1, 0, 2, 3)  # (B, T, C, D)
            return z, y
