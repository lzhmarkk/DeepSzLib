import torch
import numpy as np
import torch.nn as nn
from sklearn.svm import SVC
from models.SVM.extract_features import extract_features


class SVM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_channels = args.n_channels
        self.dim = args.input_dim
        self.preprocess = args.preprocess

        self.training_data = []
        self.training_truth = []
        self.fit = False
        self.model = []
        for c in range(self.n_channels):
            self.training_data.append([])
            self.training_truth.append([])
            self.model.append(SVC())

        # No need to backward
        args.backward = False
        args.epochs = 1

    def forward(self, x, p, y):
        # (B, T, C, D)
        bs = x.shape[0]
        device = x.device
        x = x.cpu().numpy()
        p = p.cpu().numpy()

        if self.training:
            for c in range(self.n_channels):
                _x = extract_features(x[:, :, c, :], axis=1, preprocess=self.preprocess).reshape(bs, -1)  # (B, D*H)
                self.training_data[c].extend(_x)
                self.training_truth[c].extend(p)
            return torch.from_numpy(p).to(device)

        # validate
        else:
            if not self.fit:
                for c in range(self.n_channels):
                    data = np.stack(self.training_data[c], axis=0)
                    truth = np.stack(self.training_truth[c], axis=0)
                    # data = data[torch.randperm(len(data))][:10000]
                    self.model[c].fit(data, truth)
                    print(f"Fit channel: {c}")
                self.fit = True

            z = []
            for c in range(self.n_channels):
                _x = extract_features(x[:, :, c, :], axis=1, preprocess=self.preprocess).reshape(bs, -1)  # (B, D*H)
                _z = self.model[c].predict(_x)  # (B)
                z.append(_z)

            z = torch.from_numpy(np.stack(z, axis=-1)).to(device)  # (B, C)

            # voting
            z = torch.mean(z, dim=-1).float()  # (B)
            return z
