import math
import torch
import torch.nn as nn


class Patching(nn.Module):
    def __init__(self, seq_len, patch_imple, patch_size, patch_stride, hidden, patch_bias=False,
                 instance_norm=True, eps=1e-8):
        super().__init__()

        self.patch_imple = patch_imple
        self.patch_size = patch_size  # Sz
        self.patch_stride = patch_stride  # Strd
        self.patch_bias = patch_bias
        self.hidden = hidden
        self.instance_norm = instance_norm
        self.eps = eps

        if self.patch_imple == 'fc':
            self.mapping = nn.Linear(patch_size, hidden, bias=patch_bias)
        elif self.patch_imple == 'fft':
            self.mapping = nn.Linear(patch_size // 2, hidden, bias=patch_bias)
        elif self.patch_imple == 'multiscale_fft':
            pass
        else:
            raise ValueError()

        self.n_patches = math.ceil((seq_len - self.patch_size) / self.patch_stride) + 1

    def forward(self, x):
        batch_size, seq_len = x.shape  # (B, L)
        device = x.device

        patches = []

        if self.patch_imple == 'fc':

            if self.instance_norm:
                x, _, _ = self.instance_normalize(x)

            for pid in range(self.n_patches):
                _start = self.patch_stride * pid
                _end = self.patch_stride * pid + self.patch_size
                patch = x[:, _start: _end]
                if patch.shape[1] < self.patch_size:
                    patch = torch.cat([patch, torch.zeros(batch_size, self.patch_size - patch.shape[1]).to(device)], dim=-1)

                patches.append(patch)

            patches = torch.stack(patches, dim=1)  # (B, L//Strd, Sz)
            patch_emb = self.mapping(patches)  # (B, L//Strd, H)

        elif self.patch_imple == 'fft':
            for pid in range(self.n_patches):
                _start = self.patch_stride * pid
                _end = self.patch_stride * pid + self.patch_size
                patch = x[:, _start: _end]
                if patch.shape[1] < self.patch_size:
                    patch = torch.cat([patch, torch.zeros(batch_size, self.patch_size - patch.shape[1]).to(device)], dim=-1)

                patch_fft = self.compute_FFT(patch, n=self.patch_size)  # (B, L//Strd, Sz//2)

                if self.instance_norm:
                    patch_fft, _, _ = self.instance_normalize(patch_fft)

                patches.append(patch_fft)

            patches = torch.stack(patches, dim=1)  # (B, L//Strd, Sz//2)
            patch_emb = self.mapping(patches)  # (B, L//Strd, H)

        else:
            raise ValueError()

        return patch_emb

    def instance_normalize(self, x):
        if x.ndim == 2:
            mean = x.mean(dim=1, keepdims=True)
            std = x.std(dim=1, keepdims=True)
        elif x.ndim == 3:
            mean = x.mean(dim=[1, 2], keepdims=True)
            std = x.std(dim=[1, 2], keepdims=True)
        else:
            raise ValueError()

        norm_x = (x - mean) / (std + self.eps)
        return norm_x, mean, std

    def compute_FFT(self, signals, n):
        # Compute the Fourier transform using PyTorch's fft
        fourier_signal = torch.fft.fft(signals, n=n, dim=1)

        # Only take the positive frequency part
        idx_pos = n // 2
        fourier_signal = fourier_signal[:, :idx_pos]

        # Compute the amplitude and avoid log of 0 by adding a small constant
        amp = torch.abs(fourier_signal)
        amp[amp == 0.0] = self.eps

        # Compute the log amplitude
        FT = torch.log(amp)

        return FT
