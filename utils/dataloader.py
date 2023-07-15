import os
import json
import h5py
import torch
import numpy as np
from utils.utils import Scaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler


class DataSet(Dataset):
    def __init__(self, path, name, n_samples, args):
        self.path = path
        self.n_samples = n_samples
        self.name = name
        self.argument = args.argument and name == 'train'
        self.n_samples_per_file = args.n_samples_per_file
        self.norm = args.norm
        self.scaler = args.scaler

        print(f"{self.n_samples} samples in {name} set, ", end='')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        """
        :return: u (1), x (T, C, D), y (T, C, D), label (1)
        """
        file_id = item // self.n_samples_per_file
        smp_id = item % self.n_samples_per_file

        with h5py.File(os.path.join(self.path, f"{file_id}.h5"), "r") as hf:
            x, y, l = hf['x'][()], hf['y'][()], hf['l'][()]

        x, y, l = x[smp_id], y[smp_id], l[smp_id]
        if self.norm:
            x = self.scaler.transform([x])[0]
            y = self.scaler.transform([y])[0]

        if self.argument:
            # x, flip_pairs = self.__random_flip(x)
            x = self.__random_scale(x)
        return 0, x, y, l

    def __random_flip(self, x):
        raise NotImplementedError("Deprecated since flipping is not applicable for model without graph")
        x = x.clone()

        if np.random.choice([True, False]):
            flip_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
            for pair in flip_pairs:
                x[:, [pair[0], pair[1]], :] = x[:, [pair[1], pair[0]], :]

        else:
            flip_pairs = None

        return x, flip_pairs

    def __random_scale(self, x):
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.preprocess == 'fft':
            x += np.log(scale_factor)
        else:
            x *= scale_factor
        return x


def get_sampler(args):
    return ValueError("Load-on-demand dataloader not support balance")
    ratio = args.balance
    if ratio > 0:
        # to handle imbalance dataset
        pos_percent = args.n_pos_train / args.n_train
        weight = [float(ratio) / (1 - pos_percent), 1 / pos_percent]
        weight = torch.stack([weight[int(lab.item())] for lab in label])
        return WeightedRandomSampler(weight, len(weight))
    else:
        return None


def get_dataloader(args):
    dir = f"./data/FDUSZ"

    if not args.data_loaded:
        with open(os.path.join(dir, "config.json")) as fp:
            config = json.load(fp)
            assert all([config[k] == getattr(args, k) for k in config]), \
                f"Dataset configuration is not compatible with args. Please re-run preprocess/main.py"

        with open(os.path.join(dir, "attribute.json")) as fp:
            attribute = json.load(fp)
            for k in attribute:
                setattr(args, k, attribute[k])

        assert args.n_channels == 12 and args.sample_rate == 100
        args.window = args.window * args.sample_rate
        args.horizon = args.horizon * args.sample_rate
        args.stride = args.stride * args.sample_rate
        args.seg = args.seg * args.sample_rate
        args.scaler = Scaler(args.mean, args.std, args.norm)
        args.data_loaded = True

        print(f"Mean {args.mean}, std {args.std}")
        print(f"# Samples: train {args.n_train}, # val {args.n_val}, # test {args.n_test}")

    train_set = DataSet(os.path.join(dir, 'train'), 'train', args.n_train, args)
    val_set = DataSet(os.path.join(dir, 'val'), 'val', args.n_val, args)
    test_set = DataSet(os.path.join(dir, 'test'), 'test', args.n_test, args)
    args.dataset = {'train': train_set, 'val': val_set, 'test': test_set}

    train_loader = DataLoader(train_set, args.batch_size, shuffle=args.shuffle if args.balance < 0 else None)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
