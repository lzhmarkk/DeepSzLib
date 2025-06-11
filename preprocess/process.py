import os
import json
import h5py
import math
import shutil
import numpy as np
from tqdm import tqdm
from utils import slice_samples, patching, calculate_scaler, calculate_fft_scaler, split_dataset, count_labels, Dataset, compute_FFT

import numpy as np
from tqdm import tqdm
from scipy.fft import rfft


# Helper class for memory-efficient statistics calculation
class OnlineScaler:
    """
    Computes mean and standard deviation in a single pass using Welford's algorithm.
    This avoids loading the entire dataset into memory.
    """

    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.n_samples = 0
        self.mean = np.zeros(n_channels)
        self.m2 = np.zeros(n_channels)  # Sum of squares of differences from the current mean

    def update(self, batch):
        """
        Update the scaler with a new batch of data.
        :param batch: A numpy array of shape (..., n_channels)
        """
        if batch.size == 0:
            return

        # Reshape to (N, C) to handle different input shapes
        batch = batch.reshape(-1, self.n_channels)

        n_batch_samples = batch.shape[0]
        self.n_samples += n_batch_samples

        delta = batch - self.mean
        self.mean += np.sum(delta / self.n_samples, axis=0)
        delta2 = batch - self.mean
        self.m2 += np.sum(delta * delta2, axis=0)

    def finalize(self):
        """
        Calculate the final mean and standard deviation.
        """
        if self.n_samples < 2:
            return self.mean, np.zeros(self.n_channels)

        mean = self.mean
        std = np.sqrt(self.m2 / self.n_samples)
        return mean.tolist(), std.tolist()


def slice_samples(idx, x, label, window, horizon, stride):
    split_u, split_x, split_y, split_label, split_ylabel = [], [], [], [], []

    for i in tqdm(range(len(idx)), desc="Slice samples"):
        u = idx[i]
        assert len(x[i]) == len(label[i]), "Data and label lengths must match"

        # Calculate the number of samples that will be generated
        num_samples = (len(x[i]) - window - horizon) // stride + 1
        if num_samples <= 0:
            continue

        # Pre-allocate numpy arrays to avoid appending to lists
        n_channels = x[i].shape[-1]
        _split_x = np.empty((num_samples, window, n_channels), dtype=np.float32)
        _split_y = np.empty((num_samples, horizon, n_channels), dtype=np.float32)
        _split_label = np.empty((num_samples, window), dtype=label[i].dtype)
        _split_ylabel = np.empty(num_samples, dtype=bool)

        # Fill the pre-allocated arrays
        for k in range(num_samples):
            j = k * stride
            _split_x[k] = x[i][j: j + window, :]
            _split_y[k] = x[i][j + window: j + window + horizon, :]
            _split_label[k] = label[i][j: j + window]
            _split_ylabel[k] = label[i][j + window: j + window + horizon].any()

        _split_u = np.full(num_samples, u, dtype=int)

        split_u.append(_split_u)
        split_x.append(_split_x)
        split_y.append(_split_y)
        split_label.append(_split_label)
        split_ylabel.append(_split_ylabel)

    return split_u, split_x, split_y, split_label, split_ylabel


def patching(all_samples, patch_len):
    """
    Reshapes windowed data into patches using a zero-copy view where possible.
    :param all_samples: List of arrays of shape (N, T, C), where T is window length.
    :param patch_len: int, stands for `D`.
    :return: List of arrays of shape (N, T//D, D, C).
    """
    new_x = []
    # Ensure window length is divisible by patch_len
    window_len = all_samples[0].shape[1]
    assert window_len % patch_len == 0, f"Window length {window_len} must be divisible by patch_len {patch_len}"

    for x in tqdm(all_samples, desc="Patching"):
        # Reshape creates a view, not a copy, which is highly memory efficient
        num_patches = window_len // patch_len
        patched_x = x.reshape(x.shape[0], num_patches, patch_len, x.shape[-1])
        new_x.append(patched_x)
    return new_x


def process(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, mode, ratio, dataset_path, split, channels, n_sample_per_file):
    idx = np.arange(len(all_u))
    np.random.shuffle(idx)
    all_u = [all_u[i] for i in idx]
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Load {len(all_x)} users")

    print(f"Total {np.sum([y.shape[0] for y in all_y]).item() / sample_rate} seconds records")
    print(f"Total {np.sum([(y != 0).sum() for y in all_y]).item() / sample_rate} seconds seizure records")

    # Segment samples with pre-allocation
    all_u, all_x, all_y, all_l, all_yl = slice_samples(all_u, all_x, all_y, int(window * sample_rate), int(horizon * sample_rate),
                                                       int(stride * sample_rate))
    print(f"Slice samples. max {np.max([len(_) for _ in all_x])}, "
          f"min {np.min([len(_) for _ in all_x])}, avg {np.mean([len(_) for _ in all_x])} samples for users")

    # Patching using memory-efficient reshaping
    patch_len_s = int(patch_len * sample_rate)
    all_x = patching(all_x, patch_len_s)
    all_y = patching(all_y, patch_len_s)

    # --- Online Scaler Calculation ---
    n_channels = all_x[0].shape[-1]
    raw_scaler = OnlineScaler(n_channels)
    fft_scaler = OnlineScaler(n_channels)

    # Define which data to use for scaler calculation based on mode
    train_users_count = int(len(all_x) * ratio[0])

    print("Calculating scalers on-the-fly...")
    # Iterate through data ONCE to calculate both scalers
    # This avoids creating a full FFT-transformed copy of the dataset

    if mode == 'Inductive':
        # Use only training users for scaler
        data_iterator = all_x[:train_users_count]
    else:  # 'Transductive'
        data_iterator = all_x

    for x_user in tqdm(data_iterator, desc=f"Calculating Scalers ({mode})"):
        if mode == 'Transductive':
            # Use only a portion of each user's data
            train_idx = int(len(x_user) * ratio[0])
            data_chunk = x_user[:train_idx]
        else:  # 'Inductive'
            data_chunk = x_user

        if data_chunk.size == 0:
            continue

        # 1. Update raw data scaler
        raw_scaler.update(data_chunk)
        # 2. Compute FFT on the fly and update FFT scaler
        # We compute FFT and immediately use it for stats, then discard it.
        fft_chunk = compute_FFT(data_chunk, n=patch_len_s)
        fft_scaler.update(fft_chunk)

    # Finalize statistics
    mean_raw, std_raw = raw_scaler.finalize()
    mean_fft, std_fft = fft_scaler.finalize()
    print(f"Raw Mean: {mean_raw}, Raw Std: {std_raw}")
    print(f"FFT Mean: {mean_fft}, FFT Std: {std_fft}")
    mean = {'raw': mean_raw, 'fft': mean_fft}
    std = {'raw': std_raw, 'fft': std_fft}
    input_dim = {'raw': patch_len_s, 'fft': patch_len_s // 2}

    # split train/val/test
    datasets = split_dataset(all_u, all_x, all_y, all_l, all_yl, mode, ratio)

    # statistics
    label_count, n_pos, n_users, n_pos_users = {}, {}, {}, {}
    for stage in ['train', 'val', 'test']:
        label_count[stage] = count_labels(datasets[stage].label)
        pos_idx = np.array([_.any() for _ in datasets[stage].label])
        n_pos[stage] = np.sum(pos_idx).item()
        print(f"{len(datasets[stage].u)} samples in train, {n_pos[stage]} positive")
        n_users[stage] = len(set(datasets[stage].u))
        n_pos_users[stage] = len(set(np.array(datasets[stage].u)[pos_idx]))
        print(f"{n_users[stage]} users in train, {n_pos_users[stage]} positive")

    # save
    shutil.rmtree(dataset_path, ignore_errors=True)
    os.makedirs(dataset_path, exist_ok=True)
    with open(os.path.join(dataset_path, "./config.json"), 'w') as fp:
        config = {'window': window, 'horizon': horizon, 'stride': stride, 'patch_len': patch_len,
                  "setting": mode, "split": split}
        json.dump(config, fp, indent=4)

    with open(os.path.join(dataset_path, "./attribute.json"), 'w') as fp:
        attribute = {'sample_rate': sample_rate, 'n_samples_per_file': n_sample_per_file,
                     "n_channels": len(channels), "channels": channels,
                     'n_user': len(idx), 'mean': mean, 'std': std, 'input_dim': input_dim}
        for stage in ['train', 'val', 'test']:
            attribute.update({f'n_user_{stage}': n_users[stage],
                              f"n_pos_user_{stage}": n_pos_users[stage],
                              f"n_{stage}": len(datasets[stage].u),
                              f"n_pos_{stage}": n_pos[stage],
                              f"label_count_{stage}": label_count[stage]})
        json.dump(attribute, fp, indent=4)

    for stage in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_path, stage), exist_ok=True)
        ds = datasets[stage]
        for i in tqdm(range(math.ceil(len(ds.u) / n_sample_per_file))):
            with h5py.File(os.path.join(dataset_path, stage, f"{i}.h5"), "w") as hf:
                hf.create_dataset("u", data=ds.u[i * n_sample_per_file:(i + 1) * n_sample_per_file])
                hf.create_dataset("x", data=ds.x[i * n_sample_per_file:(i + 1) * n_sample_per_file])
                hf.create_dataset("next", data=ds.next[i * n_sample_per_file:(i + 1) * n_sample_per_file])
                hf.create_dataset("label", data=ds.label[i * n_sample_per_file:(i + 1) * n_sample_per_file])
                hf.create_dataset("next_label", data=ds.next_label[i * n_sample_per_file:(i + 1) * n_sample_per_file])

    print(f"Preprocessing done")


def process_TUSZ(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, mode, dataset_path, n_sample_per_file,
                 classification=False):
    idx = np.arange(len(all_u))
    np.random.shuffle(idx)
    all_u = [all_u[i] for i in idx]
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Load {len(set(all_u))} users, {len(all_x)} files")

    # segment samples
    all_u, all_x, all_y, all_l, all_yl = slice_samples(all_u, all_x, all_y, window * sample_rate, horizon * sample_rate, stride * sample_rate)

    if classification:  # drop non-seizure samples/users for classification task
        kept_users = []
        for u in range(len(all_u)):
            idx = np.arange(len(all_u[u]))
            idx = idx[all_l[u].any(axis=1)]

            print(f"User {u} keeps {len(idx)} samples")

            # drop non-seizure samples
            if len(idx) > 0:
                all_u[u] = all_u[u][idx]
                all_x[u] = all_x[u][idx]
                all_y[u] = all_y[u][idx]
                all_l[u] = all_l[u][idx]
                all_yl[u] = all_yl[u][idx]
                kept_users.append(u)

        # drop non-seizure users
        kept_users = np.array(kept_users)
        all_u = [all_u[i] for i in kept_users]
        all_x = [all_x[i] for i in kept_users]
        all_y = [all_y[i] for i in kept_users]
        all_l = [all_l[i] for i in kept_users]
        all_yl = [all_yl[i] for i in kept_users]
        print(f"Keep {len(kept_users)} users that have seizures")

        assert all([label.any() for labels in all_l for label in labels])

    print(f"Slice samples. max {np.max([len(_) for _ in all_x])}, "
          f"min {np.min([len(_) for _ in all_x])}, avg {np.mean([len(_) for _ in all_x])} samples for users")

    # patching
    all_x = patching(all_x, int(patch_len * sample_rate))
    all_y = patching(all_y, int(patch_len * sample_rate))

    # calculate scaler
    if mode == 'train':
        mean, std = calculate_scaler(all_x, "Inductive", [1, 0, 0])
        print(f"Mean {mean}, std {std}")
        fft_x_all, (fft_mean, fft_std) = calculate_fft_scaler(all_x, "Inductive", [1, 0, 0], int(patch_len * sample_rate))
        print(f"FFT mean {fft_mean}, fft std {fft_std}")
        mean = {'raw': mean, 'fft': fft_mean}
        std = {'raw': std, 'fft': fft_std}
        input_dim = {'raw': all_x[-1].shape[2], 'fft': fft_x_all[-1].shape[2]}

    dataset = Dataset(np.concatenate(all_u), np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_l), np.concatenate(all_yl))
    print(f"{len(dataset.u)} samples in {mode}")

    # statistics
    label_count = count_labels(dataset.label)
    pos_idx = np.array([_.any() for _ in dataset.label])
    n_pos = np.sum(pos_idx).item()
    print(f"{len(pos_idx)} samples in {mode}, {n_pos} positive")
    n_users = len(set(dataset.u))
    n_pos_users = len(set(np.array(dataset.u)[pos_idx]))
    print(f"{n_users} users in {mode}, {n_pos_users} positive")

    # attribute
    attribute = {}
    attribute['n_user'] = n_users
    attribute[f'n_user_{mode}'] = n_users
    attribute[f'n_pos_user_{mode}'] = n_pos_users
    attribute[f'n_{mode}'] = len(dataset.u)
    attribute[f'n_pos_{mode}'] = np.sum([_.any() for _ in dataset.label]).item()
    attribute[f"label_count_{mode}"] = label_count
    if mode == 'train':
        attribute.update({'mean': mean, 'std': std, 'input_dim': input_dim})

    # save

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, mode), exist_ok=True)
    for i in tqdm(range(math.ceil(len(dataset.u) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, mode, f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=dataset.u[i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=dataset.x[i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next", data=dataset.next[i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("label", data=dataset.label[i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next_label", data=dataset.next_label[i * n_sample_per_file:(i + 1) * n_sample_per_file])

    print(f"Preprocessing done")
    return attribute
