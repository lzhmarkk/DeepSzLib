import math
import torch
import torch.nn as nn
from models.utils import Segmentation
from models.CrossFormer.CrossEncoder import Encoder


class CrossFormer(nn.Module):
    def __init__(self, args):
        super(CrossFormer, self).__init__()
        self.hidden = args.hidden
        self.in_len = args.window // args.seg
        self.seg = args.seg
        self.merge = args.merge
        self.preprocess = args.preprocess
        self.channels = args.n_channels
        self.enc_layer = args.enc_layer
        self.dropout = args.dropout
        self.n_heads = args.n_heads
        self.n_router = args.n_router

        # assert self.preprocess == 'seg'
        if self.preprocess == 'seg':
            self.dim = args.hidden
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        # Embedding
        self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.channels, self.in_len, self.hidden))
        self.pre_norm = nn.LayerNorm(self.hidden)

        # Encoder
        self.encoder = Encoder(e_blocks=self.enc_layer, win_size=self.merge, d_model=self.hidden, n_heads=self.n_heads,
                               d_ff=4 * self.hidden, block_depth=1, dropout=self.dropout, in_seg_num=self.in_len,
                               factor=self.n_router)

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden * (1 + self.enc_layer), self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

    def forward(self, x, p, y):
        # (B, T, C, S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x)  # (B, T, C, D)

        x = x.permute(0, 2, 1, 3)
        x += self.enc_pos_embedding  # (B, C, 1+T, D)
        x = self.pre_norm(x)

        # encoder
        enc_out = self.encoder(x)  # (B, C, T', D)[], list with different T'
        enc_out = [out.mean(dim=2) for out in enc_out]  # (B, C, D)[]

        # decoder
        enc_out = torch.cat(enc_out, dim=2)
        enc_out = enc_out.reshape(bs, -1)  # (B, C*D*L)
        enc_out = torch.tanh(enc_out)
        z = self.decoder(enc_out).squeeze(dim=-1)
        return z
