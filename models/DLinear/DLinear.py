import torch
import torch.nn as nn
from models.DLinear.utils import series_decomp


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, args):
        super(DLinear, self).__init__()
        self.seq_len = args.window
        self.pred_len = args.hidden
        assert args.preprocess != 'seg' and args.preprocess != 'fft'

        # Decompsition Kernel Size
        self.decompsition = series_decomp(args.kernel_size)
        self.individual = args.individual
        self.channels = args.n_channels

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        if args.act == 'tanh':
            self.act = nn.Tanh()
        elif args.act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        self.Linear_Channel = nn.Sequential(nn.Linear(self.pred_len * self.channels, self.pred_len),
                                            nn.GELU(),
                                            nn.Linear(self.pred_len, self.pred_len * self.channels))
        self.norm = nn.LayerNorm(self.pred_len * self.channels)
        self.fc = nn.Linear(self.pred_len * self.channels, 1)

    def forward(self, x):
        # (B, T, C, D/S)
        bs = x.shape[0]
        x = x.squeeze(dim=-1)

        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        x = self.act(x)
        x = x.reshape(bs, -1)  # (B, C*T)
        x = self.norm(self.Linear_Channel(x) + x)
        x = self.fc(x).squeeze()
        return x
