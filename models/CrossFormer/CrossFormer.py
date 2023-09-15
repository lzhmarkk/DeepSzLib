import math
import torch
import torch.nn as nn
from models.utils import Segmentation
from models.CrossFormer.CrossEncoder import Encoder
from models.CrossFormer.CrossDecoder import Decoder


class CrossFormer(nn.Module):
    def __init__(self, args):
        super(CrossFormer, self).__init__()
        self.hidden = args.hidden
        self.in_len = args.window // args.seg
        self.out_len = args.horizon // args.seg
        self.seg = args.seg
        self.merge = args.merge
        self.preprocess = args.preprocess
        self.channels = args.n_channels
        self.enc_layer = args.enc_layer
        self.dropout = args.dropout
        self.n_heads = args.n_heads
        self.n_router = args.n_router
        self.anomaly_len = args.anomaly_len

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

        self.task = args.task
        assert 'cls' in self.task or 'anomaly' in self.task
        self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden * (1 + self.enc_layer), self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        if 'pred' in self.task:
            self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.channels, self.out_len, self.hidden))
            self.predictor = Decoder(self.dim, self.enc_layer + 1, self.hidden, self.n_heads, 4 * self.hidden,
                                     self.dropout, out_seg_num=self.out_len, factor=self.n_router)

    def predict(self, enc_out):
        enc_out = torch.cat(enc_out, dim=2)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)  # (B, C*D*L)
        enc_out = torch.tanh(enc_out)
        z = self.decoder(enc_out).squeeze(dim=-1)
        return z

    def forward(self, x, p, y):
        # (B, T, C, S)
        bs = x.shape[0]
        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            x = self.fc(x)  # (B, T, C, D)

        if 'cls' in self.task:
            x = x.permute(0, 2, 1, 3)
            x = x + self.enc_pos_embedding  # (B, C, T, D)
            x = self.pre_norm(x)
            enc_out = self.encoder(x)  # (B, C, T', D)[], list with different T'
            enc_out_mean = [out.mean(dim=2) for out in enc_out]  # (B, C, D)[]
            z = self.predict(enc_out_mean)

        else:
            out = []
            for t in range(1, self.in_len + 1):
                xt = x[:, max(0, t - self.anomaly_len):t, :, :]
                xt = xt.permute(0, 2, 1, 3)
                xt = xt + self.enc_pos_embedding[:, :, -xt.shape[2]:]  # (B, C, T, D)
                xt = self.pre_norm(xt)
                enc_out = self.encoder(xt)  # (B, C, T', D)[], list with different T'
                enc_out_mean = [out.mean(dim=2) for out in enc_out]  # (B, C, D)[]
                z = self.predict(enc_out_mean)
                out.append(z)
            z = torch.stack(out, dim=1)

        if 'pred' not in self.task:
            return z, None
        else:
            dec_in = self.dec_pos_embedding.repeat(bs, 1, 1, 1)  # (B, C, T, D)
            y = self.predictor(dec_in, enc_out)
            y = y.transpose(1, 2)  # (B, T, C, D)

            return z, y
