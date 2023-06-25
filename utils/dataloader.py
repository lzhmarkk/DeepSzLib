import os
import torch
import numpy as np
from scipy.fftpack import fft
from utils.utils import Scaler
from torch.utils.data import Dataset, DataLoader


def filter(data, sigma=3):
    data = data.copy()
    mean = data.mean()
    std = data.std()
    idx_l = (data <= mean - sigma * std)
    idx_r = (data >= mean + sigma * std)
    data[idx_l | idx_r] = 0.
    return data, idx_l.sum() + idx_r.sum()


def compute_FFT(signals, n):
    """ from git@github.com:tsy935/eeg-gnn-ssl.git
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)

    return FT


class DataSet(Dataset):
    def __init__(self, args, x, y, p):

        new_x = []
        for i in range(len(x)):
            _x = x[i].reshape(args.window // args.seg, args.seg, args.n_channels).permute(0, 2, 1)  # (T, C, S)
            if args.preprocess == 'seg':
                pass
            elif args.preprocess == 'fft':
                new__x = []
                for t in range(args.window // args.seg):
                    new__x.append(torch.from_numpy(compute_FFT(_x[t, :, :].numpy(), args.seg)))
                _x = torch.stack(new__x, dim=0)
            else:
                _x = x[i].unsqueeze(dim=-1)  # (T, C, 1)

            new_x.append(_x)

        self.x = new_x
        self.y = y
        self.p = torch.stack(p)
        self.len = len(x)
        print("{:.2f}% samples are positive".format(self.p.sum() * 100 / len(self.p)))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.p[item]


class Data:
    def __init__(self, dir, args, norm=True):
        # load files from disk
        x, y = [], []
        files = os.listdir(dir)
        n_users = len(files)
        for f in files:
            data = np.load(os.path.join(dir, f))
            x.append(data['x'])  # (T, C)
            y.append(data['y'])  # (T)
            sample_rate = data['sr'].item()

        x = self.normalize(x, norm)

        # splitting samples
        args.sample_rate = sample_rate
        horizon = args.horizon * sample_rate
        window = args.window * sample_rate
        stride = args.stride * sample_rate
        split_x, split_y, split_p = [], [], []

        for u in range(n_users):

            _split_x, _split_y, _split_p = [], [], []
            for i in range(0, len(x[u]) - horizon - window, stride):
                _x = torch.from_numpy(x[u][i:i + horizon, :])
                _y = torch.from_numpy(x[u][i + horizon:i + horizon + window, :])
                _p = float(y[u][i:i + horizon].any())

                _split_x.append(_x)
                _split_y.append(_y)
                _split_p.append(_p)

            split_x.append(torch.stack(_split_x, dim=0).float())
            split_y.append(torch.stack(_split_y, dim=0).float())
            split_p.append(torch.tensor(_split_p))

        self.x = split_x
        self.y = split_y
        self.p = split_p
        self.n_users = n_users
        self.n_channels = data['x'].shape[1]
        _ = torch.cat(split_p, dim=0).flatten()
        print(f"{(_ == 1).sum() * 100 / len(_)}% samples are positive")

    def normalize(self, x, norm):
        if norm:
            _x = np.concatenate(x, axis=0)  # (sum(T), C)
            self.scaler = Scaler(_x.mean(), _x.std())
        else:
            self.scaler = Scaler(0, 1)

        x = [self.scaler.transform(_x) for _x in x]
        return x

    def balance(self, train_x, train_y, train_p):
        # re-sample
        p = torch.stack(train_p, dim=0)
        pos_idx = torch.where(p == 1)[0]
        neg_idx = torch.where(p == 0)[0]
        neg_idx = neg_idx[torch.randperm(len(neg_idx))[:len(pos_idx)]]
        idx = torch.cat([neg_idx, pos_idx], dim=0)
        idx = idx[torch.randperm(len(idx))]
        train_x = [train_x[i] for i in idx]
        train_y = [train_y[i] for i in idx]
        train_p = [train_p[i] for i in idx]
        return train_x, train_y, train_p

    def split_dataset(self, args):
        ratio = [float(r) for r in str(args.split).split('/')]
        ratio = [r / sum(ratio) for r in ratio]

        train_x, train_y, train_p = [], [], []
        val_x, val_y, val_p = [], [], []
        test_x, test_y, test_p = [], [], []
        if args.mode == 'Transductive':
            for u in range(self.n_users):
                x, y, p = self.x[u], self.y[u], self.p[u]
                train_idx = int(len(x) * ratio[0])
                val_idx = train_idx + int(len(x) * ratio[1])
                train_x.extend(x[:train_idx])
                train_y.extend(y[:train_idx])
                train_p.extend(p[:train_idx])
                val_x.extend(x[train_idx:val_idx])
                val_y.extend(y[train_idx:val_idx])
                val_p.extend(p[train_idx:val_idx])
                test_x.extend(x[val_idx:])
                test_y.extend(y[val_idx:])
                test_p.extend(p[val_idx:])
        elif args.mode == 'Inductive':
            train_idx = int(self.n_users * ratio[0])
            val_idx = int(self.n_users * ratio[1])
            for u in range(self.n_users)[:train_idx]:
                x, y, p = self.x[u], self.y[u], self.p[u]
                train_x.extend([(u, _x) for _x in x])
                train_y.extend(y)
                train_p.extend(p)
            for u in range(self.n_users)[train_idx:val_idx]:
                x, y, p = self.x[u], self.y[u], self.p[u]
                val_x.extend([(u, _x) for _x in x])
                val_y.extend(y)
                val_p.extend(p)
            for u in range(self.n_users)[val_idx:]:
                x, y, p = self.x[u], self.y[u], self.p[u]
                test_x.extend([(u, _x) for _x in x])
                test_y.extend(y)
                test_p.extend(p)
        else:
            raise ValueError(f"Not implemented mode: {args.mode}")

        train_x, train_y, train_p = self.balance(train_x, train_y, train_p)

        return (train_x, train_y, train_p), (val_x, val_y, val_p), (test_x, test_y, test_p)


def get_dataloader(args):
    dir = f"./data/FDUSZ"
    data = Data(dir, args)
    train_set, val_set, test_set = data.split_dataset(args)
    print("# Samples", len(train_set[0]), len(val_set[0]), len(test_set[0]))

    args.n_users = data.n_users
    args.n_channels = data.n_channels
    args.sample_rate = args.sample_rate
    args.seg = args.seg * args.sample_rate
    args.window = args.window * args.sample_rate
    args.horizon = args.horizon * args.sample_rate
    args.scaler = data.scaler

    train_set, val_set, test_set = DataSet(args, *train_set), DataSet(args, *val_set), DataSet(args, *test_set)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
