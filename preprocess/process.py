import os
import json
import h5py
import math
import shutil
import numpy as np
from tqdm import tqdm
from utils import slice_samples, patching, calculate_scaler, calculate_fft_scaler, split_dataset, count_labels, Dataset


def process(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, mode, ratio, dataset_path, split, channels, n_sample_per_file):
    idx = np.arange(len(all_u))
    np.random.shuffle(idx)
    all_u = [all_u[i] for i in idx]
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Load {len(all_x)} users")

    print(f"Total {np.sum([y.shape[0] for y in all_y]).item() / sample_rate} seconds records")
    print(f"Total {np.sum([(y != 0).sum() for y in all_y]).item() / sample_rate} seconds seizure records")

    # segment samples
    all_u, all_x, all_y, all_l, all_yl = slice_samples(all_u, all_x, all_y, window * sample_rate, horizon * sample_rate, stride * sample_rate)
    print(f"Slice samples. max {np.max([len(_) for _ in all_x])}, "
          f"min {np.min([len(_) for _ in all_x])}, avg {np.mean([len(_) for _ in all_x])} samples for users")

    # patching
    all_x = patching(all_x, int(patch_len * sample_rate))
    all_y = patching(all_y, int(patch_len * sample_rate))

    # calculate scaler
    mean, std = calculate_scaler(all_x, mode, ratio)
    print(f"Mean {mean}, std {std}")
    fft_x_all, (fft_mean, fft_std) = calculate_fft_scaler(all_x, mode, ratio, int(patch_len * sample_rate))
    print(f"FFT mean {fft_mean}, fft std {fft_std}")
    mean = {'raw': mean, 'fft': fft_mean}
    std = {'raw': std, 'fft': fft_std}
    input_dim = {'raw': all_x[-1].shape[2], 'fft': fft_x_all[-1].shape[2]}

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
