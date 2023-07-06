import os
import h5py
import torch
import numpy as np
from scipy.fftpack import fft
from utils.utils import Scaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler, WeightedRandomSampler

data = None


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


class DataContainer:
    def __init__(self, dir, args):
        # load files from disk
        x, label = [], []
        files = os.listdir(dir)
        if args.debug:
            files = files[:5]
        for f in files:
            with h5py.File(os.path.join(dir, f), 'r') as f:
                _x = f["x"][()]
                _label = f["y"][()]
                sample_rate = f["sr"][()]
                assert sample_rate == 100, f"Resample rate in h5 file is not {100}"
                assert _x.shape[1] == 12, f"Channel in h5 file is not {12}"
            x.append(_x)  # (T, C)
            label.append(_label)  # (T)

        assert len(x) == len(label) == len(files)
        args.n_users = len(files)
        args.n_channels = x[0].shape[1]
        args.sample_rate = sample_rate
        args.seg = args.seg * sample_rate

        # filter abnormal data
        # x, drop_rate = self.filter(x, mean, std)  # (T, C)[]
        # print("Drop abnormal data. {:.2f}% entries are dropped".format(drop_rate * 100))

        # transform (fft, etc.)
        x = self.transform(x, args)  # (T, S, C)[]
        print("Apply transformation")

        # calculate mean and std
        ratio, mean, std = self.get_distribution(x, args)
        print(f"Calculate mean and std. mean {mean}, std {std}")

        # normalize according to training data
        scaler, x = self.normalize(x, mean, std, args.norm)
        args.scaler = scaler
        print("Z-normalize")

        # segment samples
        x, y, label = self.segment_samples(x, label, args)
        self.x = x
        self.y = y
        self.label = label
        print("Segment samples")

        # split train/val/test samples
        train_set, val_set, test_set = self.split_dataset(args, ratio)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        print(f"Split train/val/test sets. # train {len(train_set[0])}, # val {len(val_set[0])}, # test {len(test_set[0])}")

        args.window = args.window * sample_rate
        args.horizon = args.horizon * sample_rate
        args.stride = args.stride * sample_rate

    def get_distribution(self, x, args):
        ratio = [float(r) for r in str(args.split).split('/')]
        ratio = [r / sum(ratio) for r in ratio]

        values, n_samples = [], []
        if args.mode == 'Transductive':
            for u in range(args.n_users):
                train_idx = int(len(x[u]) * ratio[0])
                values.append(x[u][:train_idx])
                n_samples.append(x[u][:train_idx].size)

        elif args.mode == 'Inductive':
            train_idx = int(args.n_users * ratio[0])
            for u in range(args.n_users)[:train_idx]:
                values.append(x[u])
                n_samples.append(x[u].size)

        # mean
        all_sum = 0
        for v, n in zip(values, n_samples):
            all_sum += v.sum()
        mean = all_sum / sum(n_samples)

        # std
        all_sqrt = 0
        for v, n in zip(values, n_samples):
            all_sqrt += ((v - mean) ** 2).sum()
        std = np.sqrt(all_sqrt / sum(n_samples))

        return ratio, mean, std

    def filter(self, x, mean, std):
        new_x = []
        drop_sum, all_sum = 0, 0
        for data in x:
            idx_l = data <= mean - 3 * std
            idx_r = data >= mean + 3 * std
            data[idx_l | idx_r] = 0.
            new_x.append(data)
            drop_sum += idx_l.sum() + idx_r.sum()
            all_sum += data.shape[0] * data.shape[1]
        return new_x, drop_sum / all_sum

    def transform(self, x, args):
        # preprocess
        new_x = []
        for _x in x:
            seg_x = []
            for i in range(len(_x) // args.seg):
                seg = _x[i * args.seg:(i + 1) * args.seg]  # (S, C)
                if args.preprocess == 'fft':
                    seg = compute_FFT(seg.T, n=args.seg).T
                seg_x.append(seg)
            seg_x = np.stack(seg_x, axis=0)  # (T, S, C)
            new_x.append(seg_x)
        return new_x

    def normalize(self, x, mean, std, norm):
        scaler = Scaler(mean, std, norm)
        x = scaler.transform(x)
        return scaler, x

    def segment_samples(self, x, label, args):
        # note: args.horizon, args.window and args.stride must not multiply by sample_rate
        split_x, split_y, split_label = [], [], []
        for u in range(args.n_users):
            _split_x, _split_y, _split_label = [], [], []
            for i in range(0, len(x[u]) - args.horizon - args.window, args.stride):
                _x = torch.from_numpy(x[u][i:i + args.horizon, :])
                _y = torch.from_numpy(x[u][i + args.horizon:i + args.horizon + args.window, :])
                _l = float(label[u][i * args.seg:(i + args.horizon) * args.seg].any())

                _split_x.append(_x)
                _split_y.append(_y)
                _split_label.append(_l)

            split_x.append(torch.stack(_split_x, dim=0).float())
            split_y.append(torch.stack(_split_y, dim=0).float())
            split_label.append(torch.tensor(_split_label))

        return split_x, split_y, split_label

    def split_dataset(self, args, ratio):
        train_u, train_x, train_y, train_label = [], [], [], []
        val_u, val_x, val_y, val_label = [], [], [], []
        test_u, test_x, test_y, test_label = [], [], [], []

        if args.mode == 'Transductive':
            for u in range(args.n_users):
                x, y, label = self.x[u], self.y[u], self.label[u]
                train_idx = int(len(x) * ratio[0])
                val_idx = train_idx + int(len(x) * ratio[1])
                # train
                train_u.extend([u] * train_idx)
                train_x.extend(x[:train_idx])
                train_y.extend(y[:train_idx])
                train_label.extend(label[:train_idx])
                # val
                val_u.extend([u] * (val_idx - train_idx))
                val_x.extend(x[train_idx:val_idx])
                val_y.extend(y[train_idx:val_idx])
                val_label.extend(label[train_idx:val_idx])
                # test
                test_u.extend([u] * (len(x) - val_idx))
                test_x.extend(x[val_idx:])
                test_y.extend(y[val_idx:])
                test_label.extend(label[val_idx:])

        elif args.mode == 'Inductive':
            train_idx = int(args.n_users * ratio[0])
            val_idx = int(args.n_users * ratio[1])
            for u in range(args.n_users)[:train_idx]:
                x, y, label = self.x[u], self.y[u], self.label[u]
                train_u.extend([u] * len(x))
                train_x.extend(x)
                train_y.extend(y)
                train_label.extend(label)
            for u in range(args.n_users)[train_idx:val_idx]:
                x, y, label = self.x[u], self.y[u], self.label[u]
                val_u.extend([u] * len(x))
                val_x.extend(x)
                val_y.extend(y)
                val_label.extend(label)
            for u in range(args.n_users)[val_idx:]:
                x, y, label = self.x[u], self.y[u], self.label[u]
                test_u.extend([u] * len(x))
                test_x.extend(x)
                test_y.extend(y)
                test_label.extend(label)
        else:
            raise ValueError(f"Not implemented mode: {args.mode}")

        return (train_u, train_x, train_y, torch.stack(train_label)), \
            (val_u, val_x, val_y, torch.stack(val_label)), \
            (test_u, test_x, test_y, torch.stack(test_label))


class DataSet(Dataset):
    def __init__(self, u, x, y, label, name):
        assert len(u) == len(x) == len(y) == len(label)
        self.u = torch.tensor(u)
        self.x = torch.stack(x, dim=0).transpose(3, 2)
        self.y = torch.stack(y, dim=0).transpose(3, 2)
        self.label = label
        self.name = name
        self.len = len(x)

        print(f"{self.len} samples in {name} set")
        print("{:.2f}% samples are positive".format(self.label.sum() * 100 / len(self.label)))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.u[item], self.x[item], self.y[item], self.label[item]


def get_sampler(label, ratio):
    if ratio > 0:
        # to handle imbalance dataset
        pos_percent = (label == 1).sum() / len(label)
        weight = [float(ratio) / (1 - pos_percent), 1 / pos_percent]
        weight = torch.stack([weight[int(lab.item())] for lab in label])
        return WeightedRandomSampler(weight, len(weight))
    else:
        return None


def get_dataloader(args):
    global data
    if data is None:
        dir = f"./data/FDUSZ"
        data = DataContainer(dir, args)

    train_set = DataSet(*data.train_set, 'train')
    val_set = DataSet(*data.val_set, 'val')
    test_set = DataSet(*data.test_set, 'test')

    train_loader = DataLoader(train_set, args.batch_size, sampler=get_sampler(data.train_set[3], args.balance), shuffle=args.shuffle if args.balance < 0 else None)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
