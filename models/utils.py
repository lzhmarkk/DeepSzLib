import torch
import torch.nn as nn


class Segmentation(nn.Module):
    def __init__(self, length, dim, channels, individual=False):
        super().__init__()

        self.S = length
        self.dim = dim
        self.channels = channels
        self.individual = individual

        if self.individual:
            self.linear1 = nn.ModuleList()
            self.linear2 = nn.ModuleList()
            for c in range(self.channels):
                self.linear1.append(nn.Linear(self.S, self.dim))
                self.linear2.append(nn.Linear(self.dim, self.S))
        else:
            self.linear1 = nn.Linear(self.S, self.dim)
            self.linear2 = nn.Linear(self.dim, self.S)

    def segment(self, x):
        # (B, T, C, S)
        if self.individual:
            x_emb = []
            for c in range(self.channels):
                x_emb.append(self.linear1[c](x[:, :, c, :]))
            x_emb = torch.stack(x_emb, dim=2)
        else:
            x_emb = self.linear1(x)
        return x_emb  # (B, T, C, D)

    def unsegment(self, x_emb):
        # (B, T, C, D)
        if self.individual:
            x = []
            for c in range(self.channels):
                x.append(self.linear2[c](x_emb[:, :, c, :]))
            x = torch.stack(x, dim=2)
        else:
            x = self.linear2(x_emb)
        return x  # (B, T, C, S)


def check_tasks(model):
    required_tasks = []

    for _ in model.task:
        if _ in model.unsupported_tasks:
            raise ValueError(f"Not supported task {_} for model {model.__class__.__name__}")

        if _ in model.supported_tasks:
            required_tasks.append(_)

    assert len(required_tasks) == 1, f"Only support one main task, but {required_tasks} are given."
