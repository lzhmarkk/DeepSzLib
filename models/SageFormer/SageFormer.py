import torch
import torch.nn as nn
from models.MTGNN.MTGNN import graph_constructor, mixprop


class SageFormer(nn.Module):
    """
    http://arxiv.org/abs/2307.01616
    """
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
        self.preprocess = args.preprocess
        self.n_cls_tokens = args.n_cls_tokens
        self.input_dim = args.input_dim
        self.gcn_depth = args.gcn_depth
        assert self.preprocess == 'fft'

        # fft
        self.fc = nn.Linear(self.input_dim, self.hidden)

        self.encoder = nn.ModuleList()
        self.conv = nn.ModuleList()
        for l in range(self.layers):
            self.encoder.append(nn.TransformerEncoderLayer(self.hidden, self.heads, 4 * self.hidden, self.dropout, batch_first=True))
            if l > 0:
                self.conv.append(mixprop(self.hidden, self.hidden, self.gcn_depth, self.dropout, 0))

        self.decoder = nn.Sequential(nn.Linear(self.n_cls_tokens * self.channels * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        self.gc = graph_constructor(self.channels, self.hidden, 1)
        self.cls_token = nn.Parameter(torch.randn(1, self.n_cls_tokens, self.channels, self.hidden), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn([self.n_cls_tokens + self.window // self.seg, self.channels, self.hidden]), requires_grad=True)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]
        x = self.fc(x)  # (B, T, C, D)

        x = torch.cat([self.cls_token.expand(bs, -1, -1, -1), x], dim=1)
        x = x + self.pos_emb.unsqueeze(0)  # (B, n+T, C, D)
        x = x.transpose(2, 1)  # (B, C, n+T, D)
        x = x.reshape(bs * self.channels, self.n_cls_tokens + self.window // self.seg, self.hidden)  # (B*C, n+T, D)

        g = self.gc(x)
        h = self.encoder[0](x)
        for l in range(1, self.layers):
            cls, h = h[:, :self.n_cls_tokens], h[:, self.n_cls_tokens:]  # (B*C, n, D),  # (B*C, T, D)
            cls = cls.reshape(bs, self.channels, self.n_cls_tokens, self.hidden).permute(0, 3, 1, 2)  # (B, D, C, n)
            cls = self.conv[l - 1](cls, g)
            cls = cls.permute(0, 2, 3, 1).reshape(bs * self.channels, self.n_cls_tokens, self.hidden)  # (B*C, n, D)
            h = torch.cat([cls, h], dim=1)
            h = self.encoder[l](h)  # (B*C, n+T, D)

        # decoder
        z = h[:, :self.n_cls_tokens, :]  # (B*C, n, D)
        z = z.reshape(bs, self.channels * self.n_cls_tokens * self.hidden)  # (B, C*n*D)

        z = torch.tanh(z)
        z = self.decoder(z).squeeze(dim=-1)  # (B)

        return z
