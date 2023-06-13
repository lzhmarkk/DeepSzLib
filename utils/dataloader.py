import os
import mne
import numpy as np
from torch.utils.data import Dataset, DataLoader


def filter(data, sigma=3):
    data = data.copy()
    mean = data.mean()
    std = data.std()
    idx_l = (data <= mean - sigma * std)
    idx_r = (data >= mean + sigma * std)
    data[idx_l | idx_r] = 0.
    return data, idx_l.sum() + idx_r.sum()


def load_data(edf_path, txt_path):
    data = mne.io.read_raw_edf(edf_path, preload=True)
    data = data.get_data().T  # (T, C)

    timestamps = []
    truth = np.zeros([data.shape[0]], dtype=int)
    with open(txt_path, 'r') as fp:
        lines = fp.readlines()

    for i, l in enumerate(lines):
        words = str(l).split('\t')
        assert (i % 2 == 0 and '开始' in words[0]) or (i % 2 == 1 and '结束' in words[0])
        timestamps.append(words[1])
    timestamps = list(zip(timestamps[0::2], timestamps[1::2]))

    for timestamp in timestamps:
        s_time, e_time = timestamp
        s_time = [int(_) for _ in s_time.split(':')]
        e_time = [int(_) for _ in e_time.split(':')]
        s_time = s_time[0] * 3600 + s_time[1] * 60 + s_time[2]
        e_time = e_time[0] * 3600 + e_time[1] * 60 + e_time[2]
        s_time *= 500
        e_time *= 500
        truth[s_time:e_time] = 1

    return data, truth


class DataSet(Dataset):
    def __init__(self, x, y, p):
        self.x = x
        self.y = y
        self.p = p
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.p[item]


class Data:
    def __init__(self, dir, args):
        files = os.listdir(dir)
        files = list(set([_[:-4] for _ in files]))
        files = sorted(files)

        # load files
        x = []
        y = []
        for f in files:
            edf_path = os.path.join(dir, f + ".edf")
            txt_path = os.path.join(dir, f + ".txt")

            data, truth = load_data(edf_path, txt_path)

            # sifting in each channel
            for i in range(data.shape[1]):
                _data, _ = filter(data[:, i], sigma=args.sigma)
                data[:, i] = _data

            x.append(data)
            y.append(truth)

        # splitting samples
        split_x, split_y, split_p = [], [], []
        for u in range(len(files)):
            _split_x, _split_y, _split_p = [], [], []
            for i in range(0, len(x[u]) - args.horizon - args.window, args.step):
                _split_x.append(x[u][i:i + args.horizon, :])
                _split_y.append(x[u][i + args.horizon:i + args.horizon + args.window, :])
                _split_p.append(y[u][i:i + args.horizon])
            split_x.append(_split_x)
            split_y.append(_split_y)
            split_p.append(_split_p)

        self.x = split_x
        self.y = split_y
        self.p = split_p
        self.n_users = len(files)
        self.n_channels = data.shape[-1]

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
                train_x.extend([(u, _x) for _x in x[:train_idx]])
                train_y.extend(y[:train_idx])
                train_p.extend(p[:train_idx])
                val_x.extend([(u, _x) for _x in x[train_idx:val_idx]])
                val_y.extend(y[train_idx:val_idx])
                val_p.extend(p[train_idx:val_idx])
                test_x.extend([(u, _x) for _x in x[val_idx:]])
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
                x, y = self.x[u], self.y[u]
                val_x.extend([(u, _x) for _x in x])
                val_y.extend(y)
                val_p.extend(p)
            for u in range(self.n_users)[val_idx:]:
                x, y = self.x[u], self.y[u]
                test_x.extend([(u, _x) for _x in x])
                test_y.extend(y)
                test_p.extend(p)
        else:
            raise ValueError(f"Not implemented mode: {args.mode}")

        return DataSet(train_x, train_y, train_p), DataSet(val_x, val_y, val_p), DataSet(test_x, test_y, test_p)


def get_dataloader(args):
    dir = f"./data/edf_noName_SeizureFile"
    data = Data(dir, args)
    train_set, val_set, test_set = data.split_dataset(args)
    args.n_users = data.n_users
    args.n_channels = data.n_channels

    train_loader = DataLoader(train_set, args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
