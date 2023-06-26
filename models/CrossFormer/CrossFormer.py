import math
import torch
import torch.nn as nn
from models.utils import Segmentation
from models.CrossFormer.CrossEncoder import Encoder


class Crossformer(nn.Module):
    def __init__(self, args):
        super(Crossformer, self).__init__()
        self.hidden = args.hidden
        self.in_len = args.window // args.seg
        self.out_len = 1
        self.seg = args.seg
        self.merge = args.merge
        self.preprocess = args.preprocess
        self.channels = args.n_channels
        self.enc_layer = args.enc_layer
        self.dropout = args.dropout
        self.n_heads = args.n_heads
        self.n_router = args.n_router

        assert self.preprocess == 'seg'
        self.dim = args.hidden

        # Embedding
        self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.channels, self.in_len, self.hidden))
        self.pre_norm = nn.LayerNorm(self.hidden)

        # Encoder
        self.encoder = Encoder(e_blocks=self.enc_layer, win_size=self.merge, d_model=self.hidden, n_heads=self.n_heads,
                               d_ff=4 * self.hidden, block_depth=1, dropout=self.dropout, in_seg_num=self.in_len, factor=self.n_router)

        t, length = self.in_len * 2, self.in_len
        for l in range(1, self.enc_layer):
            length = math.ceil(length / self.merge)
            t += length
        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden * t, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

    def forward(self, x):
        # (B, T, C, S)
        bs = x.shape[0]

        x = self.segmentation.segment(x)  # (B, T, C, S) -> (B, T, C, D)
        x = x.permute(0, 2, 1, 3)

        x += self.enc_pos_embedding  # (B, C, T, D)
        x = self.pre_norm(x)

        # encoder
        enc_out = self.encoder(x)  # (B, C, T', D)[], list with different T'

        # decoder
        enc_out = torch.cat(enc_out, dim=2)
        enc_out = enc_out.reshape(bs, -1)  # (B, C*sum(T')*D)
        z = self.decoder(enc_out).squeeze()
        return z
