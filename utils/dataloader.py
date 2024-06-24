import os
import json
import math
import h5py
import torch
import numpy as np
from utils.utils import Scaler
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler, WeightedRandomSampler
from preprocess.utils import compute_FFT, get_sample_label


def handle_label(l, task, seq_len, patch_len):
    if 'onset_detection' in task:
        return l.reshape(len(l), seq_len, patch_len).any(axis=2)
    elif 'detection' in task:
        return l.any(axis=1)
    elif 'classification' in task:
        l = [get_sample_label(_) for _ in l]
        assert all(l)
        l = [_ - 1 for _ in l]  # There is no non-seizure sample
        return l
    else:
        raise ValueError()


class DataSet(Dataset):
    def __init__(self, path, name, args):
        self.path = path
        self.name = name
        self.argument = args.argument and name == 'train'
        self.n_samples_per_file = args.n_samples_per_file
        self.norm = not args.no_norm
        self.scaler = args.scaler
        self.preprocess = args.preprocess
        self.patch_len = args.patch_len
        self.seq_len = args.window // self.patch_len
        self.channels = args.n_channels
        self.pin_memory = args.pin_memory
        self.task = args.task

        self.n_samples = getattr(args, f"n_{name}")
        self.n_pos = getattr(args, f"n_pos_{name}")
        self.label_count = getattr(args, f"label_count_{name}")
        print(f"{self.n_samples} samples in {name} set, {100 * self.n_pos / self.n_samples}% are positive")

        self.data = {'u': [], 'x': [], 'y': [], 'l': []}
        for file_id in range(math.ceil(self.n_samples / self.n_samples_per_file)):
            with h5py.File(os.path.join(self.path, f"{file_id}.h5"), "r") as hf:
                if self.pin_memory:
                    u, x, y = hf['u'][:], hf['x'][:], hf['next'][:]
                    self.data['u'].append(u)
                    self.data['x'].append(x)
                    self.data['y'].append(y)

                # must be loaded in advance for a balanced sampler
                l = handle_label(hf['label'][:], self.task, self.seq_len, self.patch_len)
                self.data['l'].append(l)

        for k in self.data:
            if len(self.data[k]) > 0:
                self.data[k] = np.concatenate(self.data[k])
        assert self.n_samples == len(self.data['l'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, indices):
        """
        :return: u (*), x (*, T, C, D), y (*, T, C, D), label (*)
        `*` refers to the length of indices
        """
        y = None
        if self.pin_memory:
            assert np.max(indices) < self.n_samples
            smp_ids = sorted(indices)
            u = self.data['u'][smp_ids]
            x = self.data['x'][smp_ids]
            l = self.data['l'][smp_ids]
            if 'prediction' in self.task:
                y = self.data['y'][smp_ids]
        else:
            # all indices must belong to a single .h5 file
            file_ids = [idx // self.n_samples_per_file for idx in indices]
            smp_ids = [idx % self.n_samples_per_file for idx in indices]
            assert len(set(file_ids)) == 1, f"Batched h5 loader error"
            file_id = file_ids[0]
            smp_ids = sorted(smp_ids)

            with h5py.File(os.path.join(self.path, f"{file_id}.h5"), "r") as hf:
                u = hf['u'][smp_ids]
                x = hf['x'][smp_ids]
                l = hf['label'][smp_ids]
                l = handle_label(l, self.task, self.seq_len, self.patch_len)
                if 'prediction' in self.task:
                    y = hf['next'][smp_ids]

        x = x.transpose(0, 1, 3, 2)
        if y is not None:
            y = y.transpose(0, 1, 3, 2)

        x, y = self.apply_argumentation(x, y)

        x, y = self.apply_transformation(x, y)

        x, y = self.apply_normalization(x, y)

        return u, x, y, l

    def apply_argumentation(self, x, y):
        if self.argument:
            x = self._random_scale(x)
            x = self._random_noise(x, self.scaler.mean, self.scaler.std)
            x = self._random_smooth(x, 0.2)
            if y is not None:
                y = self._random_scale(y)
                y = self._random_noise(y, self.scaler.mean, self.scaler.std)
                y = self._random_smooth(y, 0.2)

        return x, y

    def apply_transformation(self, x, y):
        B, T, C, _ = x.shape
        if self.preprocess == 'raw':
            x[x >= self.scaler.mean + 3 * self.scaler.std] = 0
            x[x <= -self.scaler.mean - 3 * self.scaler.std] = 0
        elif self.preprocess == 'fft':
            x = x.reshape(B * T * C, self.patch_len)
            x = compute_FFT(x, n=self.patch_len)
            x = x.reshape(B, T, C, self.patch_len // 2)
            if y is not None:
                y = y.reshape(B * T * C, self.patch_len)
                y = compute_FFT(y, n=self.patch_len)
                y = y.reshape(B, T, C, self.patch_len // 2)
        else:
            pass

        return x, y

    def apply_normalization(self, x, y):
        if self.norm:
            x = self.scaler.transform(x)
            if y is not None:
                y = self.scaler.transform(y)

        return x, y

    def _random_scale(self, x):
        scale_factor = np.random.uniform(0.8, 1.2)
        x *= scale_factor
        return x

    def _random_noise(self, x, mean, std):
        noise = np.random.normal(0, np.sqrt(0.1 * np.var(x)), x.shape)
        x = x + noise
        return x

    def _random_smooth(self, x, p):
        length = self.seq_len * self.patch_len
        x = x.copy()
        x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], length, self.channels)

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                idx = np.random.choice(range(1, length - 1), int(p * length), replace=False)
                x[i, idx, j] = (x[i, idx - 1, j] + x[i, idx + 1, j]) / 2

        x = x.reshape(x.shape[0], self.seq_len, self.patch_len, self.channels).transpose(0, 1, 3, 2)
        return x


class CollectFn:
    def __call__(self, data):
        u, x, y, l = [], [], [], []
        for sample in data:
            u.append(sample[0])
            x.append(sample[1])
            y.append(sample[2])
            l.append(sample[3])

        u = torch.from_numpy(np.concatenate(u, axis=0)).int()
        x = torch.from_numpy(np.concatenate(x, axis=0)).float()

        if y[0] is not None:
            y = torch.from_numpy(np.concatenate(y, axis=0)).float()
        else:
            y = None
        if l[0] is not None:
            l = torch.from_numpy(np.concatenate(l, axis=0)).float()
        else:
            l = None

        return u, x, y, l


class BatchSamplerX(BatchSampler):
    def __init__(self, dataset, args, mode):
        if args.balance > 0 and mode == 'train':
            label_weight = {int(k): len(dataset) / v for k, v in dataset.label_count.items()}
            sample_weights = []
            for lab in dataset.data['l'].astype(int):
                if 'classification' in args.task:
                    sample_weights.append(label_weight[lab + 1])
                else:
                    sample_weights.append(label_weight[lab])

            sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
            print(f"Balanced sampler. {label_weight}")
        else:
            if args.shuffle and mode == 'train':
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        super().__init__(sampler, args.batch_size, drop_last=False)
        self.n_samples_per_file = args.n_samples_per_file
        self.pin_memory = args.pin_memory

    def split_batch_by_file(self, batch):
        batch = list(set(batch))

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


def get_dataloader(args):
    print(f"Use {args.setting} setting")

    if not args.data_loaded:
        with open(os.path.join(args.data_folder, "config.json")) as fp:
            config = json.load(fp)
            assert all([config[k] == getattr(args, k) for k in config]), \
                f"Dataset configuration is not compatible with args. Please re-run preprocess/main.py"

        with open(os.path.join(args.data_folder, "attribute.json")) as fp:
            attribute = json.load(fp)
            for k in attribute:
                setattr(args, k, attribute[k])

        all_classes = set()
        all_classes.update(args.label_count_train.keys())
        all_classes.update(args.label_count_val.keys())
        all_classes.update(args.label_count_test.keys())
        args.n_classes = 1 if len(all_classes) <= 2 else len(all_classes)
        print(f"Total {len(all_classes)} classes")

        args.window = int(args.window * args.sample_rate)
        args.horizon = int(args.horizon * args.sample_rate)
        args.stride = int(args.stride * args.sample_rate)
        args.patch_len = int(args.patch_len * args.sample_rate)
        args.scaler = Scaler(args.mean[args.preprocess], args.std[args.preprocess], not args.no_norm)
        args.input_dim = int(args.input_dim[args.preprocess])
        args.data_loaded = True

        print(f"Mean {args.mean}, std {args.std}")
        print(f"# Samples: train {args.n_train}, # val {args.n_val}, # test {args.n_test}")

    train_set = DataSet(os.path.join(args.data_folder, 'train'), 'train', args)
    val_set = DataSet(os.path.join(args.data_folder, 'val'), 'val', args)
    test_set = DataSet(os.path.join(args.data_folder, 'test'), 'test', args)
    args.data = {'train': train_set, 'val': val_set, 'test': test_set}

    collate_fn = CollectFn()
    train_sampler = BatchSamplerX(train_set, args, mode='train')
    val_sampler = BatchSamplerX(val_set, args, mode='val')
    test_sampler = BatchSamplerX(test_set, args, mode='test')

    n_worker = args.n_worker if not args.pin_memory else 0
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=n_worker, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=n_worker, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=n_worker, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
