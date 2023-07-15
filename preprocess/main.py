import os
import mne
import json
import h5py
import math
import numpy as np
from tqdm import tqdm
from scipy.signal import resample
from utils import slice_samples, segmentation, calculate_scaler, calculate_fft_scaler, split_dataset

dir = f"./data/"
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'T3', 'T4', 'EKG', 'EMG']
sample_rate = 500
n_sample_per_file = 1000
np.random.seed(0)


def load_edf_data(edf_path, sample_rate, resample_rate):
    data = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')

    data_channels = []
    for c in channels:
        matched_channels = sorted(list(filter(lambda e: c in e, data.ch_names)))
        matched_channel = matched_channels[0]

        data_channels.append(data[matched_channel][0])

    data = np.concatenate(data_channels, axis=0).T  # (T, C)
    assert data.shape[1] == 12

    resample_data = resample(data, num=data.shape[0] // sample_rate * resample_rate, axis=0)
    return data, resample_data


def load_txt_data(txt_path, length, sample_rate):
    truth = np.zeros([length], dtype=float)

    if txt_path is None:
        return truth

    timestamps = []
    with open(txt_path, 'r') as fp:
        lines = fp.readlines()

    s = False
    for i, l in enumerate(lines):
        words = str(l).split('\t')
        if (i % 2 == int(s) and '开始' in words[0]) or (i % 2 == int(not s) and '结束' in words[0]):
            timestamps.append(words[1])
        else:
            s = not s
    timestamps = list(zip(timestamps[0::2], timestamps[1::2]))

    for timestamp in timestamps:
        s_time, e_time = timestamp
        s_time = [int(_) for _ in s_time.split(':')]
        e_time = [int(_) for _ in e_time.split(':')]
        s_time = s_time[0] * 3600 + s_time[1] * 60 + s_time[2]
        e_time = e_time[0] * 3600 + e_time[1] * 60 + e_time[2]
        s_time *= sample_rate
        e_time *= sample_rate
        assert 0 <= s_time and e_time <= length, f"{txt_path}"
        truth[s_time:e_time] = 1

    return truth


if __name__ == '__main__':
    with open("./preprocess/config.json", 'r') as fp:
        config = json.load(fp)
        resample_rate = config["resample_rate"]
        preprocess = config["preprocess"]
        mode = config["mode"]
        split = config["split"]
        ratio = [float(r) for r in str(split).split('/')]
        ratio = [r / sum(ratio) for r in ratio]
        window = config["window"]
        horizon = config["horizon"]
        stride = config["stride"]
        seg = config["seg"]

    # load data
    user_id = 0
    all_x, all_y = [], []
    patient_dir = os.path.join(dir, 'edf_noName_SeizureFile')
    patient_files = sorted(list(set([_[:-4] for _ in os.listdir(patient_dir)])))
    for f in tqdm(patient_files, desc="Loading patient"):
        _, x = load_edf_data(os.path.join(patient_dir, f + ".edf"), sample_rate, resample_rate)
        y = load_txt_data(os.path.join(patient_dir, f + ".txt"), length=x.shape[0], sample_rate=resample_rate)
        all_x.append(x)
        all_y.append(y)
        user_id += 1

    control_dir = os.path.join(dir, 'control')
    control_files = sorted(list(set([_[:-4] for _ in os.listdir(control_dir)])))
    for f in tqdm(control_files, desc="Loading control"):
        _, x = load_edf_data(os.path.join(control_dir, f + ".edf"), sample_rate, resample_rate)
        y = load_txt_data(None, length=x.shape[0], sample_rate=resample_rate)
        all_x.append(x)
        all_y.append(y)
        user_id += 1

    print(f"Load {len(all_x)} users")
    sample_rate = resample_rate

    # shuffle users
    idx = np.arange(len(all_x))
    np.random.shuffle(idx)
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Shuffle users, {idx}")

    # segment samples
    all_x, all_y, all_l = slice_samples(all_x, all_y, window * sample_rate, horizon * sample_rate, stride * sample_rate)
    print(f"Slice samples. max {np.max([len(_) for _ in all_x])}, "
          f"min {np.min([len(_) for _ in all_x])}, avg {np.mean([len(_) for _ in all_x])} samples for users")

    # segmentation
    all_x = segmentation(all_x, seg * sample_rate)
    all_y = segmentation(all_y, seg * sample_rate)

    # calculate scaler
    mean, std = calculate_scaler(all_x, mode, ratio)
    print(f"Mean {mean}, std {std}")
    fft_x_all, (fft_mean, fft_std) = calculate_fft_scaler(all_x, mode, ratio, seg * sample_rate)
    print(f"FFT mean {fft_mean}, fft std {fft_std}")
    mean = {'seg': mean, 'fft': fft_mean}
    std = {'seg': std, 'fft': fft_std}
    input_dim = {'seg': all_x[-1].shape[2], 'fft': fft_x_all[-1].shape[2]}

    # split train/val/test
    train_set, val_set, test_set = split_dataset(all_x, all_y, all_l, mode, ratio)
    print(f"{len(train_set[0])} samples in train, {len(val_set[0])} samples in validate, "
          f"and {len(test_set[0])} samples in test.")

    # save
    dataset_path = os.path.join(dir, 'FDUSZ')
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'test'), exist_ok=True)

    with open(os.path.join(dataset_path, "./config.json"), 'w') as fp:
        config = {'window': window, 'horizon': horizon, 'stride': stride, 'seg': seg,
                  "mode": mode, "split": split}
        json.dump(config, fp, indent=2)

    with open(os.path.join(dataset_path, "./attribute.json"), 'w') as fp:
        n_pos_train = np.sum(train_set[2]) / len(train_set[2])
        n_pos_val = np.sum(val_set[2]) / len(val_set[2])
        n_pos_test = np.sum(test_set[2]) / len(test_set[2])
        n_pos = (np.sum(train_set[2]) + np.sum(val_set[2]) + np.sum(test_set[2])) / (
                len(train_set[2]) + len(val_set[2]) + len(test_set[2]))
        attribute = {'sample_rate': sample_rate, 'n_samples_per_file': n_sample_per_file, "n_channels": 12,
                     'n_train': len(train_set[0]), 'n_val': len(val_set[0]), 'n_test': len(test_set[0]),
                     'n_pos_train': n_pos_train, 'n_pos_val': n_pos_val, 'n_pos_test': n_pos_test,
                     'mean': mean, 'std': std, 'input_dim': input_dim}
        json.dump(attribute, fp, indent=2)

    for i in tqdm(range(math.ceil(len(train_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'train', f"{i}.h5"), "w") as hf:
            hf.create_dataset("x", data=train_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("y", data=train_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("l", data=train_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])

    for i in tqdm(range(math.ceil(len(val_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'val', f"{i}.h5"), "w") as hf:
            hf.create_dataset("x", data=val_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("y", data=val_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("l", data=val_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])

    for i in tqdm(range(math.ceil(len(test_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'test', f"{i}.h5"), "w") as hf:
            hf.create_dataset("x", data=test_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("y", data=test_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("l", data=test_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])

    print(f"Preprocessing done")
