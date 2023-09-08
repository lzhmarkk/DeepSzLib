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

        self.gc = graph_constructor(self.channels, self.hidden, 1)
        self.cls_token = nn.Parameter(torch.randn(1, self.n_cls_tokens, self.channels, self.hidden), requires_grad=True)
        self.pos_emb = nn.Parameter(torch.randn([self.n_cls_tokens + self.window // self.seg, self.channels, self.hidden]), requires_grad=True)

        self.task = args.task
        assert 'pred' not in self.task
        assert 'cls' in self.task or 'anomaly' in self.task
        if 'cls' in self.task:
            self.decoder = nn.Sequential(nn.Linear(self.n_cls_tokens * self.channels * self.hidden, self.hidden),
                                         nn.GELU(),
                                         nn.Linear(self.hidden, 1))
        else:
            self.decoder = nn.Sequential(nn.Linear(self.channels * self.hidden, self.hidden),
                                         nn.GELU(),
                                         nn.Linear(self.hidden, 1))

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]
        x = self.fc(x)  # (B, T, C, D)

        x = torch.cat([x, self.cls_token.expand(bs, -1, -1, -1)], dim=1)
        x = x + self.pos_emb.unsqueeze(0)  # (B, T+n, C, D)
        x = x.transpose(2, 1)  # (B, C, T+n, D)
        x = x.reshape(bs * self.channels, self.window // self.seg + self.n_cls_tokens, self.hidden)  # (B*C, T+n, D)
        mask = torch.triu(torch.ones(self.window // self.seg + self.n_cls_tokens, self.window // self.seg + self.n_cls_tokens, device=x.device) \
                          * float('-inf'), diagonal=1).float()
        mask[-self.n_cls_tokens:] = 0.

        g = self.gc(x)
        h = self.encoder[0](x, src_mask=mask)
        for l in range(1, self.layers):
            cls, h = h[:, -self.n_cls_tokens:], h[:, :-self.n_cls_tokens]  # (B*C, n, D),  # (B*C, T, D)
            cls = cls.reshape(bs, self.channels, self.n_cls_tokens, self.hidden).permute(0, 3, 1, 2)  # (B, D, C, n)
            cls = self.conv[l - 1](cls, g)
            cls = cls.permute(0, 2, 3, 1).reshape(bs * self.channels, self.n_cls_tokens, self.hidden)  # (B*C, n, D)
            h = torch.cat([h, cls], dim=1)
            h = self.encoder[l](h, src_mask=mask)  # (B*C, T+n, D)

        # decoder
        if 'cls' in self.task:
            z = h[:, -self.n_cls_tokens:, :]  # (B*C, n, D)
            z = z.reshape(bs, self.channels * self.n_cls_tokens * self.hidden)  # (B, C*n*D)

            z = torch.tanh(z)
            z = self.decoder(z).squeeze(dim=-1)  # (B)
        else:
            z = h[:, :-self.n_cls_tokens, :]  # (B*C, T, D)
            z = z.reshape(bs, self.channels, self.window // self.seg, self.hidden).transpose(1, 2)  # (B, C, T, D)
            z = z.reshape(bs, self.window // self.seg, self.channels * self.hidden)  # (B, T, C*D)
            z = torch.tanh(z)
            z = self.decoder(z).squeeze(dim=-1)  # (B, T)

        return z, None
