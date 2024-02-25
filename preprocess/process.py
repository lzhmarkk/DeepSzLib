import os
import json
import h5py
import math
import numpy as np
from tqdm import tqdm
from utils import slice_samples, segmentation, calculate_scaler, calculate_fft_scaler, split_dataset


def process(all_u, all_x, all_y, sample_rate, window, horizon, stride, seg, mode, ratio, dataset_path, split, channels, n_sample_per_file):
    idx = np.arange(len(all_u))
    np.random.shuffle(idx)
    all_u = [all_u[i] for i in idx]
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Load {len(all_x)} users")

    print(f"Total {np.sum([y.shape[0] for y in all_y]).item() / sample_rate} seconds records")
    print(f"Total {np.sum([y.sum() for y in all_y]).item() / sample_rate} seconds seizure records")

    # segment samples
    all_u, all_x, all_y, all_l, all_yl = slice_samples(all_u, all_x, all_y, window * sample_rate, horizon * sample_rate, stride * sample_rate)
    print(f"Slice samples. max {np.max([len(_) for _ in all_x])}, "
          f"min {np.min([len(_) for _ in all_x])}, avg {np.mean([len(_) for _ in all_x])} samples for users")

    # segmentation
    all_x = segmentation(all_x, int(seg * sample_rate))
    all_y = segmentation(all_y, int(seg * sample_rate))

    # calculate scaler
    mean, std = calculate_scaler(all_x, mode, ratio)
    print(f"Mean {mean}, std {std}")
    fft_x_all, (fft_mean, fft_std) = calculate_fft_scaler(all_x, mode, ratio, int(seg * sample_rate))
    print(f"FFT mean {fft_mean}, fft std {fft_std}")
    mean = {'raw': mean, 'fft': fft_mean}
    std = {'raw': std, 'fft': fft_std}
    input_dim = {'raw': all_x[-1].shape[2], 'fft': fft_x_all[-1].shape[2]}

    # split train/val/test
    train_set, val_set, test_set = split_dataset(all_u, all_x, all_y, all_l, all_yl, mode, ratio)

    # statistics
    pos_idx_train = np.array([_.any() for _ in train_set[3]])
    pos_idx_val = np.array([_.any() for _ in val_set[3]])
    pos_idx_test = np.array([_.any() for _ in test_set[3]])
    n_pos_train = np.sum(pos_idx_train).item()
    n_pos_val = np.sum(pos_idx_val).item()
    n_pos_test = np.sum(pos_idx_test).item()
    print(f"{len(train_set[0])} samples in train, {n_pos_train} positive")
    print(f"{len(val_set[0])} samples in validate, {n_pos_val} positive")
    print(f"{len(test_set[0])} samples in test, {n_pos_test} positive")
    n_users_train = len(set(train_set[0]))
    n_users_val = len(set(val_set[0]))
    n_users_test = len(set(test_set[0]))
    n_pos_users_train = len(set(np.array(train_set[0])[pos_idx_train]))
    n_pos_users_val = len(set(np.array(val_set[0])[pos_idx_val]))
    n_pos_users_test = len(set(np.array(test_set[0])[pos_idx_test]))
    print(f"{n_users_train} users in train, {n_pos_users_train} positive")
    print(f"{n_users_val} users in validate, {n_pos_users_val} positive")
    print(f"{n_users_test} users in test, {n_pos_users_test} positive")

    # save
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'test'), exist_ok=True)

    with open(os.path.join(dataset_path, "./config.json"), 'w') as fp:
        config = {'window': window, 'horizon': horizon, 'stride': stride, 'seg': seg,
                  "setting": mode, "split": split}
        json.dump(config, fp, indent=2)

    with open(os.path.join(dataset_path, "./attribute.json"), 'w') as fp:
        n_pos = n_pos_train + n_pos_val + n_pos_test
        attribute = {'sample_rate': sample_rate, 'n_samples_per_file': n_sample_per_file,
                     "n_channels": len(channels), "channels": channels,
                     'n_user': len(idx), 'n_user_train': n_users_train, 'n_user_val': n_users_val, 'n_user_test': n_users_test,
                     'n_pos_user_train': n_pos_users_train, 'n_pos_user_val': n_pos_users_val, 'n_pos_user_test': n_pos_users_test,
                     'n_train': len(train_set[0]), 'n_val': len(val_set[0]), 'n_test': len(test_set[0]),
                     'n_pos_train': n_pos_train, 'n_pos_val': n_pos_val, 'n_pos_test': n_pos_test,
                     'mean': mean, 'std': std, 'input_dim': input_dim}
        json.dump(attribute, fp, indent=2)

    # train
    for i in tqdm(range(math.ceil(len(train_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'train', f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=train_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=train_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next", data=train_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("label", data=train_set[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next_label", data=train_set[4][i * n_sample_per_file:(i + 1) * n_sample_per_file])
    with h5py.File(os.path.join(dataset_path, 'train', f"label.h5"), "w") as hf:
        hf.create_dataset("labels", data=[_.any() for _ in train_set[3]])

    # val
    for i in tqdm(range(math.ceil(len(val_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'val', f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=val_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=val_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next", data=val_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("label", data=val_set[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next_label", data=val_set[4][i * n_sample_per_file:(i + 1) * n_sample_per_file])
    with h5py.File(os.path.join(dataset_path, 'val', f"label.h5"), "w") as hf:
        hf.create_dataset("labels", data=[_.any() for _ in val_set[3]])

    for i in tqdm(range(math.ceil(len(test_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'test', f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=test_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=test_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next", data=test_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("label", data=test_set[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next_label", data=test_set[4][i * n_sample_per_file:(i + 1) * n_sample_per_file])
    with h5py.File(os.path.join(dataset_path, 'test', f"label.h5"), "w") as hf:
        hf.create_dataset("labels", data=[_.any() for _ in test_set[3]])

    print(f"Preprocessing done")


def process_TUSZ(all_u, all_x, all_y, sample_rate, window, horizon, stride, seg, mode, dataset_path, n_sample_per_file, attribute):
    idx = np.arange(len(all_u))
    np.random.shuffle(idx)
    all_u = [all_u[i] for i in idx]
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Load {len(set(all_u))} users, {len(all_x)} files")

    # segment samples
    all_u, all_x, all_y, all_l, all_yl = slice_samples(all_u, all_x, all_y, window * sample_rate, horizon * sample_rate, stride * sample_rate)
    print(f"Slice samples. max {np.max([len(_) for _ in all_x])}, "
          f"min {np.min([len(_) for _ in all_x])}, avg {np.mean([len(_) for _ in all_x])} samples for users")

    # segmentation
    all_x = segmentation(all_x, int(seg * sample_rate))
    all_y = segmentation(all_y, int(seg * sample_rate))

    # calculate scaler
    if mode == 'train':
        mean, std = calculate_scaler(all_x, "Inductive", [1, 0, 0])
        print(f"Mean {mean}, std {std}")
        fft_x_all, (fft_mean, fft_std) = calculate_fft_scaler(all_x, "Inductive", [1, 0, 0], int(seg * sample_rate))
        print(f"FFT mean {fft_mean}, fft std {fft_std}")
        mean = {'raw': mean, 'fft': fft_mean}
        std = {'raw': std, 'fft': fft_std}
        input_dim = {'raw': all_x[-1].shape[2], 'fft': fft_x_all[-1].shape[2]}

    dataset = np.concatenate(all_u), np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_l), np.concatenate(all_yl)
    print(f"{len(dataset[0])} samples in {mode}")

    # statistics
    pos_idx = np.array([_.any() for _ in dataset[3]])
    n_pos = np.sum(pos_idx).item()
    print(f"{len(pos_idx)} samples in {mode}, {n_pos} positive")
    n_users = len(set(dataset[0]))
    n_pos_users = len(set(np.array(dataset[0])[pos_idx]))
    print(f"{n_users} users in {mode}, {n_pos_users} positive")

    # save
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, mode), exist_ok=True)

    # attribute
    attribute['n_user'] = attribute['n_user'] + n_users if 'n_user' in attribute else n_users
    attribute[f'n_user_{mode}'] = n_users
    attribute[f'n_pos_user_{mode}'] = n_pos_users
    attribute[f'n_{mode}'] = len(dataset[0])
    attribute[f'n_pos_{mode}'] = np.sum([_.any() for _ in dataset[3]]).item()
    if mode == 'train':
        attribute.update({'mean': mean, 'std': std, 'input_dim': input_dim})

    # data
    for i in tqdm(range(math.ceil(len(dataset[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, mode, f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=dataset[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=dataset[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next", data=dataset[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("label", data=dataset[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("next_label", data=dataset[4][i * n_sample_per_file:(i + 1) * n_sample_per_file])
    with h5py.File(os.path.join(dataset_path, mode, f"label.h5"), "w") as hf:
        hf.create_dataset("labels", data=[_.any() for _ in dataset[3]])

    print(f"Preprocessing done")
    return attribute
