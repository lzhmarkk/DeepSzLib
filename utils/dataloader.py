import os
import json
import math
import h5py
import torch
import numpy as np
from utils.utils import Scaler
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler, WeightedRandomSampler
from preprocess.utils import compute_FFT


class DataSet(Dataset):
    def __init__(self, path, name, args):
        self.path = path
        self.name = name
        self.argument = args.argument and name == 'train'
        self.n_samples_per_file = args.n_samples_per_file
        self.norm = not args.no_norm
        self.scaler = args.scaler
        self.preprocess = args.preprocess
        self.seg = args.seg
        self.seq_len = args.window // self.seg
        self.channels = args.n_channels
        self.pin_memory = args.pin_memory

        with h5py.File(os.path.join(path, f"label.h5"), "r") as hf:
            self.labels = hf['labels'][:]

        if self.name == 'train':
            self.n_samples = args.n_train
            self.n_pos = args.n_pos_train
        elif self.name == 'val':
            self.n_samples = args.n_val
            self.n_pos = args.n_pos_val
        elif self.name == 'test':
            self.n_samples = args.n_test
            self.n_pos = args.n_pos_test

        self.n_neg = self.n_samples - self.n_pos
        assert len(self.labels) == self.n_samples
        assert np.sum(self.labels).item() == self.n_pos
        print(f"{self.n_samples} samples in {name} set, {100 * self.n_pos / self.n_samples}% are positive")

        if self.pin_memory:
            self.data = {'u': [], 'x': [], 'y': [], 'l': []}
            for file_id in range(math.ceil(self.n_samples / self.n_samples_per_file)):
                with h5py.File(os.path.join(self.path, f"{file_id}.h5"), "r") as hf:
                    u, x, y, l = hf['u'][:], hf['x'][:], hf['next'][:], hf['label'][:].any(axis=1)
                    self.data['u'].append(u)
                    self.data['x'].append(x)
                    self.data['y'].append(y)
                    self.data['l'].append(l)
            for k in self.data:
                self.data[k] = np.concatenate(self.data[k])
            assert self.n_samples == len(self.data['u'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, indices):
        """
        :return: u (*), x (*, T, C, D), y (*, T, C, D), label (*)
        `*` refers to the length of indices
        """
        if self.pin_memory:
            assert np.max(indices) < self.n_samples
            smp_ids = sorted(indices)
            u, x, y, l = self.data['u'][smp_ids], self.data['x'][smp_ids], self.data['y'][smp_ids], self.data['l'][smp_ids]
        else:
            # all indices must belong to a single .h5 file
            file_ids = [idx // self.n_samples_per_file for idx in indices]
            smp_ids = [idx % self.n_samples_per_file for idx in indices]
            assert len(set(file_ids)) == 1, f"Batched h5 loader error"
            file_id = file_ids[0]
            smp_ids = sorted(smp_ids)

            with h5py.File(os.path.join(self.path, f"{file_id}.h5"), "r") as hf:
                u, x, y, l = hf['u'][smp_ids], hf['x'][smp_ids], hf['next'][smp_ids], hf['label'][smp_ids].any(axis=1)

        x = x.transpose(0, 1, 3, 2)
        y = y.transpose(0, 1, 3, 2)

        if self.argument:
            # scale
            x = self._random_scale(x)
            y = self._random_scale(y)
            # jitter
            x = self._random_noise(x, self.scaler.mean, self.scaler.std)
            y = self._random_noise(y, self.scaler.mean, self.scaler.std)
            # smooth
            x = self._random_smooth(x, 0.2)
            y = self._random_smooth(y, 0.2)

        B, T, C, _ = x.shape
        if self.preprocess == 'seg':
            pass
        elif self.preprocess == 'fft':
            x = x.reshape(B * T * C, self.seg)
            y = y.reshape(B * T * C, self.seg)
            x = compute_FFT(x, n=self.seg)
            y = compute_FFT(y, n=self.seg)
            x = x.reshape(B, T, C, self.seg // 2)
            y = y.reshape(B, T, C, self.seg // 2)
        else:
            pass

        if self.norm:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)

        # return u, x, y, l, channel_idx
        return u, x, y, l

    def _random_scale(self, x):
        scale_factor = np.random.uniform(0.8, 1.2)
        x *= scale_factor
        return x

    def _random_noise(self, x, mean, std):
        noise = np.random.normal(0, np.sqrt(0.1 * np.var(x)), x.shape)
        x = x + noise
        return x

    def _random_smooth(self, x, p):
        length = self.seq_len * self.seg
        x = x.copy()
        x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], length, self.channels)

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                idx = np.random.choice(range(1, length - 1), int(p * length), replace=False)
                x[i, idx, j] = (x[i, idx - 1, j] + x[i, idx + 1, j]) / 2

        x = x.reshape(x.shape[0], self.seq_len, self.seg, self.channels).transpose(0, 1, 3, 2)
        return x


class CollectFn:
    def __call__(self, data):
        u, x, y, l = [], [], [], []
        for sample in data:
            u.append(sample[0])
            x.append(sample[1])
            y.append(sample[2])
            l.append(sample[3])

        u = np.concatenate(u, axis=0)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        l = np.concatenate(l, axis=0)

        return torch.from_numpy(u).int(), torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(l).float()


class BatchSamplerX(BatchSampler):
    def __init__(self, dataset, batch_size, n_samples_per_file, balance, shuffle, pin_memory):
        if balance > 0:
            weight = [balance * dataset.n_pos / dataset.n_samples, dataset.n_neg / dataset.n_samples]
            weights = [weight[lab] for lab in dataset.labels.astype(int)]
            sampler = WeightedRandomSampler(weights, num_samples=(balance + 1) * dataset.n_pos, replacement=False)
            print(f"Balanced sampler. {weight}")
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        super().__init__(sampler, batch_size, drop_last=False)
        self.n_samples_per_file = n_samples_per_file
        self.pin_memory = pin_memory

    def split_batch_by_file(self, batch):
        if self.pin_memory:  # since all data in memory already, no need to split
            return [batch]

        bins = {}
        for idx in batch:
            file_id = idx // self.n_samples_per_file
            if file_id not in bins:
                bins[file_id] = []
            bins[file_id].append(idx)

        return list(bins.values())

    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0

        for idx in self.sampler:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield self.split_batch_by_file(batch)
                idx_in_batch = 0
                batch = [0] * self.batch_size

        if idx_in_batch > 0:
            yield self.split_batch_by_file(batch[:idx_in_batch])


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
    dir = f"./data/{args.dataset}" + '-' + args.setting
    print(f"Use {args.setting} setting")

    if not args.data_loaded:
        with open(os.path.join(dir, "config.json")) as fp:
            config = json.load(fp)
            assert all([config[k] == getattr(args, k) for k in config]), \
                f"Dataset configuration is not compatible with args. Please re-run preprocess/main.py"

        with open(os.path.join(dir, "attribute.json")) as fp:
            attribute = json.load(fp)
            for k in attribute:
                setattr(args, k, attribute[k])

        # assert args.n_channels == 12 and args.sample_rate == 100
        args.window = int(args.window * args.sample_rate)
        args.horizon = int(args.horizon * args.sample_rate)
        args.stride = int(args.stride * args.sample_rate)
        args.seg = int(args.seg * args.sample_rate)
        args.scaler = Scaler(args.mean[args.preprocess], args.std[args.preprocess], not args.no_norm)
        args.input_dim = int(args.input_dim[args.preprocess])
        args.data_loaded = True

        print(f"Mean {args.mean}, std {args.std}")
        print(f"# Samples: train {args.n_train}, # val {args.n_val}, # test {args.n_test}")

    train_set = DataSet(os.path.join(dir, 'train'), 'train', args)
    val_set = DataSet(os.path.join(dir, 'val'), 'val', args)
    test_set = DataSet(os.path.join(dir, 'test'), 'test', args)
    args.data = {'train': train_set, 'val': val_set, 'test': test_set}

    collate_fn = CollectFn()
    train_sampler = BatchSamplerX(train_set, args.batch_size, args.n_samples_per_file, args.balance, args.shuffle, args.pin_memory)
    val_sampler = BatchSamplerX(val_set, args.batch_size, args.n_samples_per_file, -1, False, args.pin_memory)
    test_sampler = BatchSamplerX(test_set, args.batch_size, args.n_samples_per_file, -1, False, args.pin_memory)

    n_worker = args.n_worker if not args.pin_memory else 0
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=n_worker, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=n_worker, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=n_worker, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
