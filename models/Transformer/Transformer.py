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
        self.multi_task = args.multi_task

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        if self.position_encoding:
            self.pos_emb = nn.Parameter(torch.randn([1 + self.window // self.seg, self.channels, self.hidden]), requires_grad=True)

        transformer_layer = nn.TransformerEncoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
        self.encoder = nn.TransformerEncoder(transformer_layer, self.layers)

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.channels, self.hidden), requires_grad=True)

        if self.multi_task:
            self.pred_pos_emb = nn.Parameter(torch.randn([self.horizon // self.seg, self.channels, self.hidden]), requires_grad=True)

            transformer_decoder_layer = nn.TransformerDecoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
            self.pred_decoder = nn.TransformerDecoder(transformer_decoder_layer, self.layers)
            self.pred_fc = nn.Linear(self.hidden, self.dim)

    def forward(self, x):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x)  # (B, T, C, D)

        x = torch.cat([self.cls_token.expand(bs, -1, -1, -1), x], dim=1)

        if self.position_encoding:
            x = x + self.pos_emb.unsqueeze(0)  # (B, 1+T, C, D)

        x = x.permute(1, 0, 2, 3).reshape(1 + self.window // self.seg, bs * self.channels, self.hidden)  # (1+T, B*C, D)
        h = self.encoder(x)  # (1+T, B*C, D)

        # decoder
        z = h[0, :, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.hidden)  # (B, C*D)

        z = torch.tanh(z)
        z = self.decoder(z).squeeze()  # (B)

        if not self.multi_task:
            return z
        else:
            m = h[1:, :, :]  # (T, B*C, D)
            y = self.pred_pos_emb.unsqueeze(dim=1).repeat(1, bs, 1, 1).reshape(self.horizon // self.seg, bs * self.channels, self.hidden)
            y = self.pred_decoder(y, m)
            y = y.reshape(self.horizon // self.seg, bs, self.channels, self.hidden)  # (T, B, C, D)
            y = self.pred_fc(y)
            y = y.permute(1, 0, 2, 3)  # (B, T, C, D)
            return z, y
