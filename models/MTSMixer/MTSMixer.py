import torch
import torch.nn as nn
import torch.nn.functional as fn
from models.utils import Segmentation


class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channel)
        ]) if individual else nn.Linear(seq_len, pred_len)
        # self.dropouts = nn.ModuleList()
        self.individual = individual

    def forward(self, x):
        # x: [B, T, N]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)

        else:
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, T, N]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, T, N]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class MTSMixer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.channels = args.n_channels
        self.seq_len = args.window // args.seg
        self.pred_len = self.hidden = args.hidden
        self.n_layers = args.n_layers
        self.n_nodes = self.channels
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.fac_T = args.fac_T
        self.fac_C = args.fac_C
        self.sampling = args.sampling
        self.norm = args.norm
        self.individual = args.individual
        self.preprocess = args.preprocess
        self.seg = args.seg

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        self.mlp_blocks = nn.ModuleList([MixerBlock(self.seq_len, self.n_nodes, self.d_model, self.d_ff,
                                                    self.fac_T, self.fac_C, self.sampling, self.norm)
                                         for _ in range(self.n_layers)])
        self.norm = nn.LayerNorm(self.n_nodes) if self.norm else None
        self.projection = ChannelProjection(self.seq_len, self.pred_len, self.n_nodes, self.individual)
        # self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.refine = MLPBlock(configs.pred_len, configs.d_model) if configs.refine else None

        self.decoder = nn.Sequential(nn.Linear(self.dim * self.channels * self.pred_len, self.pred_len),
                                     nn.ReLU(),
                                     nn.Linear(self.pred_len, 1))

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.permute(0, 3, 1, 2)  # (B, D, T, C)
        x = x.reshape(bs * self.dim, *x.shape[-2:])  # (B*D, T, C)

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)  # (B*D, C, T)

        x = x.reshape(bs, self.dim * self.channels * self.pred_len)
        x = self.decoder(x).squeeze(-1)

        return x
