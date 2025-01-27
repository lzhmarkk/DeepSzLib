import torch
import torch.nn as nn
import torch.nn.functional as F
from .Autoformer_EncDec import series_decomp_multi, series_decomp
from .Embed import *
from .MultiWaveletCorrelation import MultiWaveletTransform
from .FourierCorrelation import FourierBlock
from .Autoformer_EncDec import *
from .AutoCorrelation import AutoCorrelationLayer
from models.utils import check_tasks


class FEDFormer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    supported_tasks = ['detection', 'onset_detection', 'classification']
    unsupported_tasks = ['prediction']

    def __init__(self, args):
        super(FEDFormer, self).__init__()
        self.transform = args.transform
        self.mode_select = args.mode_select
        self.modes = args.n_modes
        self.seq_len = args.window // args.patch_len + 1
        self.preprocess = args.preprocess
        self.d_model = args.hidden
        self.embed = args.embed
        self.freq = args.freq
        self.enc_in = args.input_dim
        self.dropout = args.dropout
        self.base = args.base
        self.n_layers = args.n_layers
        self.moving_avg = args.moving_avg
        self.channels = args.n_channels
        self.onset_history_len = args.onset_history_len
        self.task = args.task
        check_tasks(self)

        # Decomp
        kernel_size = self.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(self.enc_in, self.d_model, self.embed, self.freq,
                                                   self.dropout)

        if self.transform == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=1, base=self.base)
        else:
            encoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        # Encoder
        enc_modes = int(min(self.modes, self.seq_len // 2))
        print('enc_modes: {}'.format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model, 8),

                    self.d_model,
                    self.d_model,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation='gelu'
                ) for _ in range(self.n_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )
        self.decoder = nn.Sequential(nn.Linear(self.channels * self.d_model, self.d_model),
                                     nn.GELU(),
                                     nn.Linear(self.d_model, args.n_classes))
        self.cls = nn.Parameter(torch.randn(1, self.channels, 1, self.enc_in))

    def predict(self, x):
        bs = x.shape[0]
        x = torch.cat([self.cls.repeat(bs, 1, 1, 1), x.transpose(1, 2)], dim=2)
        x = x.reshape(x.shape[0] * self.channels, -1, self.enc_in)

        # enc
        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (B*C, T, D)

        z = torch.mean(enc_out, dim=1)
        z = z.reshape(bs, self.channels * self.d_model)
        z = self.decoder(z).squeeze(-1)
        return z

    def forward(self, x, p, y):
        # (B, T, C, D)
        if 'detection' in self.task or 'classification' in self.task:
            z = self.predict(x)

        elif 'onset_detection' in self.task:
            out = []
            for t in range(1, self.seq_len + 1):
                xt = x[:, max(0, t - self.onset_history_len):t, :, :]
                z = self.predict(xt)
                out.append(z)
            z = torch.stack(out, dim=1)
        else:
            raise NotImplementedError

        return {'prob': z}
