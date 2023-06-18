import torch
import torch.nn as nn


class Segmentation(nn.Module):
    def __init__(self, length, dim, channels):
        super().__init__()

        self.S = length
        self.dim = dim
        self.channels = channels

        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        for c in range(self.channels):
            self.linear1.append(nn.Linear(self.S, self.dim))
            self.linear2.append(nn.Linear(self.dim, self.S))

    def segment(self, x):
        B, T, C = x.shape
        x = x.reshape(B, -1, self.S, C).permute(0, 1, 3, 2)  # (B, T', C, S)

        x_emb = []
        for c in range(self.channels):
            x_emb.append(self.linear1[c](x[:, :, c, :]))  # (B, T', D)
        x_emb = torch.stack(x_emb, dim=2)
        return x_emb  # (B, T', C, D)

    def unsegment(self, x_emb):
        B, T, C, D = x_emb.shape

        x = []
        for c in range(self.channels):
            x.append(self.linear2[c](x_emb[:, :, c, :]))  # (B, T', S)
        x = torch.stack(x, dim=2)  # (B, T', C, S)
        x = x.permute(0, 1, 3, 2).reshape(B, -1, C)
        return x
