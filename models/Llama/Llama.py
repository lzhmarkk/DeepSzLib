import torch
import torch.nn as nn
from .Patching import Patching
from .LlamaEncoder import LlamaEncoder


class Llama(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seg = args.seg
        self.window = args.window
        self.horizon = args.horizon
        self.hidden = args.hidden
        self.layers = args.layers
        self.n_channels = args.n_channels
        self.preprocess = args.preprocess
        assert args.preprocess == 'raw' and not args.norm, "Use patching_imple to change preprocessing"

        self.patching = Patching(args.window, args.patch_imple, args.patch_size, args.patch_stride, args.hidden, args.patch_bias,
                                 args.instance_norm)
        self.seq_len = self.patching.n_patches

        self.encoder = LlamaEncoder(hidden_size=self.hidden,
                                    num_hidden_layers=args.layers,
                                    num_attention_heads=args.num_attention_heads,
                                    max_position_embeddings=self.seq_len,
                                    intermediate_size=args.intermediate_size,
                                    hidden_act=args.hidden_act
                                    )

        self.decoder = nn.Linear(self.n_channels * self.hidden, 1)

    def forward(self, x, p, y):
        # (B, T, C, S)
        batch_size = x.shape[0]
        device = x.device

        x = x.transpose(1, 2).reshape(batch_size * self.n_channels, self.window)  # (B * C, T')
        x = self.patching(x)  # (B * C, T, D)

        h = self.encoder(x)  # (B * C, T, D) * n_layers

        z = h[-1][:, -1, :]  # (B * C, D)
        z = z.reshape(batch_size, self.n_channels * self.hidden)  # (B, C * D)

        z = self.decoder(z).squeeze(dim=-1)  # (B)

        return z, None
