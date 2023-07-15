import torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor
from models.SVM.extract_features import extract_features


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

        self.data = []
        self.fit = False
        self.clf = []
        for c in range(self.channels):
            self.data.append([])
            self.clf.append(LocalOutlierFactor(n_neighbors=self.n_neighbor, algorithm=self.algo, novelty=True))

        # No need to backward
        args.backward = False

    def forward(self, x, y, p):
        # (B, T, C, D)
        bs = x.shape[0]
        device = x.device

        x = x.cpu().numpy()
        p = p.cpu().numpy()

        if self.training:
            for c in range(self.channels):
                _x = extract_features(x[:, :, c, :], axis=1).reshape(bs, -1)  # (B, D*H)
                self.data[c].extend(_x)
            return torch.from_numpy(p).to(device)

        else:
            if not self.fit:
                for c in range(self.channels):
                    data = np.stack(self.data[c], axis=0)
                    data = data[torch.randperm(len(data))][:10000]
                    self.clf[c].fit(data)
                    print(f"Fit channel: {c}")
                self.fit = True

            y = []
            for c in range(self.channels):
                _x = extract_features(x[:, :, c, :], axis=1).reshape(bs, -1)  # (B, D*H)
                _y = self.clf[c].predict(_x)  # (B)
                y.append(_y)
            y = np.stack(y, axis=-1)
            y = torch.from_numpy(y).to(x.device)  # (B, C)

            # voting
            y = torch.mean(y, dim=-1).float()  # (B)
            return y
