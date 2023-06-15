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
        T, B, C = x.shape
        x = x.reshape(self.S, -1, B, C).permute(1, 2, 3, 0)  # (T', B, C, S)

        x_emb = []
        for c in range(self.channels):
            x_emb.append(self.linear1[c](x[:, :, c, :]))  # (T', B, D)
        x_emb = torch.stack(x_emb, dim=2)
        return x_emb

    def unsegment(self, x_emb):
        T, B, C, D = x_emb.shape

        x = []
        for c in range(self.channels):
            x.append(self.linear2[c](x_emb[:, :, c, :]))  # (T', B, S)
        x = torch.stack(x, dim=2)  # (T', B, C, S)
        x = x.premute(3, 0, 1, 2).reshape(-1, B, C)
        return x
