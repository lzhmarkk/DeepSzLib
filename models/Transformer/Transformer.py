import torch
import torch.nn as nn
from models.utils import Segmentation


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.window = args.window * args.sample_rate
        self.hidden = args.hidden
        self.layers = args.layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.dropout = args.dropout
        self.position_encoding = args.pos_enc

        if self.seg > 0:
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
            if self.position_encoding:
                self.pos_emb = nn.Embedding(self.window // self.seg, self.hidden)
        else:
            self.dim = 1
            if self.position_encoding:
                self.pos_emb = nn.Embedding(self.window, self.hidden)

        transformer_layer = nn.TransformerEncoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout)
        self.encoder = nn.TransformerEncoder(transformer_layer, self.layers)

        self.decoder = nn.Sequential(nn.Linear(self.channels * self.dim, self.hidden),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden, 1))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden), requires_grad=True)

    def forward(self, x):
        # (B), (B, T, C)
        u, x = x
        bs = x.shape[0]

        if self.seg > 0:
            x = self.segmentation.segment(x)  # (B, T', C, D)
        else:
            x = x.unsqueeze(dim=-1)  # (B, T', C, 1)

        x = x.permute(0, 2, 1, 3).reshape(bs * self.channels, -1, self.dim)  # (B*C, T', D)

        if self.position_encoding:
            x = x + self.pos_emb.weight.unsqueeze(0)  # (B*C, T', D)

        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1).permute(1, 0, 2)  # (1+T', B*C, D)
        z = self.encoder(x).permute(1, 0, 2)  # (B*C, 1+T', D)
        z = z[:, 0, :]  # (B*C, D)
        z = z.reshape(bs, self.channels * self.dim)  # (B, C*D)

        z = self.decoder(z).squeeze()  # (B)
        return z
