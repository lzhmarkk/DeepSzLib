import torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor


class LOF(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_neighbor = args.n_neighbor
        self.window = args.window
        self.algo = args.algo
        self.emb = args.emb
        self.channels = args.n_channels
        self.filter_rate = args.filter_rate
        self.seg = args.seg

        self._ = nn.Parameter(torch.ones([1]))  # placeholder, useless

        self.data = []
        self.fit = False
        self.clf = []
        for c in range(self.channels):
            self.data.append([])
            self.clf.append(LocalOutlierFactor(n_neighbors=self.n_neighbor, algorithm=self.algo, novelty=True))

        # No need to backward
        args.backward = False

    def forward(self, x):
        # (B, T, C, D)
        bs = x.shape[0]

        x_emb = x.permute(0,2,1,3).reshape(bs, self.channels, -1)  # (B, C, T*D)
        x_emb = x_emb.cpu().numpy()

        if self.training:
            for c in range(self.channels):
                self.data[c].extend(x_emb[:, c, :])

            return torch.zeros([x.shape[0]], device=x.device).float()

        else:
            if not self.fit:
                for c in range(self.channels):
                    data = np.array(self.data[c])
                    data = data[torch.randperm(len(data))][:10000]
                    self.clf[c].fit(data)
                    print(f"Fit channel: {c}")
                self.fit = True

            y = []
            for c in range(self.channels):
                _y = self.clf[c].predict(x_emb[:, c, :])  # (B)
                y.append(_y)
            y = np.stack(y, axis=-1)
            y = torch.from_numpy(y).to(x.device)  # (B, C)

            # voting
            y = torch.sum(y, dim=-1)  # (B)
            y[y > 0] = 0
            y[y < 0] = 1
            return y.float()
