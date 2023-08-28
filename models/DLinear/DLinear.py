import torch
import torch.nn as nn
from models.DLinear.utils import series_decomp
from models.utils import Segmentation


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, args):
        super(DLinear, self).__init__()
        self.seq_len = args.window // args.seg
        self.pred_len = 32
        # assert args.preprocess != 'seg' and args.preprocess != 'fft'
        self.preprocess = args.preprocess
        self.seg = args.seg

        # Decompsition Kernel Size
        self.decompsition = series_decomp(args.kernel_size)
        self.individual = args.individual
        self.channels = args.n_channels

        if self.preprocess == 'seg':
            self.dim = self.pred_len
            self.segmentation = Segmentation(self.seg, self.dim, self.channels)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.decoder = nn.Sequential(nn.Linear(self.pred_len * self.dim, self.pred_len),
                                     nn.GELU(),
                                     nn.Linear(self.pred_len, 1))
        self.Linear_Dim = nn.Linear(self.dim, self.dim)
        self.Linear_Channel = nn.Linear(self.channels, 1)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]
        x = x.squeeze(dim=-1)

        # x: [Batch, Input length, Channel]
        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)

        x = self.Linear_Dim(x)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 3, 1), trend_init.permute(0, 2, 3, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.dim, self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.dim, self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output  # (B, C, D, T)
        x = self.decoder(x.reshape(bs, self.channels, -1)).squeeze(-1)  # (B, C)
        x = self.Linear_Channel(x).squeeze(-1)  # (B)
        return x, None
