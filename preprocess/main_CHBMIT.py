import os
import re
import mne
import math
import h5py
import json
import numpy as np
from tqdm import tqdm
from scipy.signal import resample
from utils import slice_samples, segmentation, calculate_scaler, calculate_fft_scaler, split_dataset

origin_dir = f"./data/original_dataset/CHBMIT/1.0.0"
dest_dir = f"./data/CHBMIT/"
channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8",
            "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"]
sample_rate = 500
n_sample_per_file = 1000
np.random.seed(0)


def load_edf_data(edf_path, sample_rate, resample_rate):
    data = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')

    new_data = []
    for c in channels:
        if c in data.ch_names:
            new_data.append(data[c][0])
        elif c + '-0' in data.ch_names:
            new_data.append(data[c + '-0'][0])
        else:
            # print(f"Channel {c} not in {edf_path}")
            new_data.append(np.zeros([1, len(data)]))

    new_data = np.concatenate(new_data, axis=0).T  # (T, C)
    resample_data = resample(new_data, num=new_data.shape[0] // sample_rate * resample_rate, axis=0)
    return resample_data


def load_txt_data(extracted_info, length, sample_rate):
    truth = np.zeros([length], dtype=float)
    # print(extracted_info)

    timestamps = extracted_info['timestamp']
    assert len(timestamps) == 2 * int(extracted_info['Number of Seizures in File'])

    for i in range(0, len(timestamps), 2):
        s_time = int(timestamps[i].strip().split(' ')[0])
        e_time = int(timestamps[i + 1].strip().split(' ')[0])
        s_time *= sample_rate
        e_time *= sample_rate
        assert 0 <= s_time and e_time <= length, f"{extracted_info['File Name']}"
        truth[s_time:e_time] = 1

    return truth


def load_summary(txt_path):
    with open(txt_path, 'r') as fp:
        content = fp.read()
        blocks = content.strip().split('\n\n')

        extracted_info = {}
        current_channels = None
        for block in blocks:
            if block.startswith('Data Sampling Rate'):
                sample_rate = int(re.findall(r'Data Sampling Rate: (\d*) Hz', block)[0])
            elif block.startswith('Channel'):
                channels_in_block = []
                for line in block.split('\n'):
                    match = re.findall(r'Channel (\d{1,2})?: ([A-Za-z\d]+-[A-Za-z\d]+)', line)
                    if match:
                        channels_in_block.append(match[0][1])

                channels_in_block = set(channels_in_block)
                if current_channels is None:
                    current_channels = channels_in_block

            elif block.startswith('File Name'):
                info_dict = {}
                lines = block.strip().split('\n')
                timestamps = []
                for line in lines:
                    key, value = line.split(': ', 1)
                    if re.findall(r"Seizure(.*)Time", key):
                        timestamps.append(value.strip())
                    info_dict[key.strip()] = value.strip()

                assert current_channels is not None
                info_dict['timestamp'] = timestamps
                info_dict['Channels'] = current_channels
                extracted_info[info_dict['File Name']] = info_dict

    return sample_rate, extracted_info


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
    all_x, all_y, all_channels = [], [], []
    patient_dirs = list(filter(lambda p: 'chb' in p and os.path.isdir(os.path.join(origin_dir, p)), os.listdir(origin_dir)))
    for patient_dir in patient_dirs:
        user_id = str(patient_dir[-2:])
        print(patient_dir)
        _all_x, _all_y, _all_channels = [], [], []
        patient_dir = os.path.join(origin_dir, patient_dir)
        sample_rate, extracted_info = load_summary(os.path.join(patient_dir, f"chb{user_id}-summary.txt"))

        edf_files = list(filter(lambda f: os.path.splitext(f)[1] == '.edf', os.listdir(patient_dir)))
        for edf_file in tqdm(edf_files):
            # edf_file=f"{edf_file.split('.')[0]}.edf"
            x = load_edf_data(os.path.join(patient_dir, edf_file), sample_rate, resample_rate)

            if edf_file in extracted_info:
                y = load_txt_data(extracted_info[edf_file], length=x.shape[0], sample_rate=resample_rate)
            else:
                y = np.zeros([x.shape[0]], dtype=float)

            invalid_channels = np.all(x == 0, axis=0)
            if invalid_channels.any():
                print(f"File {edf_file} misses {np.sum(invalid_channels)} channels {np.array(channels)[invalid_channels]}")

            if (~invalid_channels).any():
                _all_x.append(x)
                _all_y.append(y)
            else:
                print(f"Skip file {edf_file}")

        # trimming
        _all_x = [x[:x.shape[0] // (100 * window) * (100 * window)] for x in _all_x]
        _all_y = [y[:y.shape[0] // (100 * window) * (100 * window)] for y in _all_y]
        all_x.append(np.concatenate(_all_x, axis=0))
        all_y.append(np.concatenate(_all_y, axis=0))

    print(f"Load {len(all_x)} users")
    sample_rate = resample_rate

    # shuffle users
    idx = np.arange(len(all_x))
    np.random.shuffle(idx)
    all_x = [all_x[i] for i in idx]
    all_y = [all_y[i] for i in idx]
    print(f"Shuffle users, {idx}")

    # segment samples
    all_u, all_x, all_y, all_l = slice_samples(idx, all_x, all_y, window * sample_rate, horizon * sample_rate, stride * sample_rate)
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
    train_set, val_set, test_set = split_dataset(all_u, all_x, all_y, all_l, mode, ratio)
    print(f"{len(train_set[0])} samples in train, {len(val_set[0])} samples in validate, "
          f"and {len(test_set[0])} samples in test.")

    # save
    dataset_path = dest_dir
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
                     'n_user': len(idx), 'n_train': len(train_set[0]), 'n_val': len(val_set[0]),
                     'n_test': len(test_set[0]),
                     'n_pos_train': n_pos_train, 'n_pos_val': n_pos_val, 'n_pos_test': n_pos_test,
                     'mean': mean, 'std': std, 'input_dim': input_dim}
        json.dump(attribute, fp, indent=2)

    for i in tqdm(range(math.ceil(len(train_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'train', f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=train_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=train_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("y", data=train_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("l", data=train_set[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])

    for i in tqdm(range(math.ceil(len(val_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'val', f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=val_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=val_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("y", data=val_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("l", data=val_set[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])

    for i in tqdm(range(math.ceil(len(test_set[0]) / n_sample_per_file))):
        with h5py.File(os.path.join(dataset_path, 'test', f"{i}.h5"), "w") as hf:
            hf.create_dataset("u", data=test_set[0][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("x", data=test_set[1][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("y", data=test_set[2][i * n_sample_per_file:(i + 1) * n_sample_per_file])
            hf.create_dataset("l", data=test_set[3][i * n_sample_per_file:(i + 1) * n_sample_per_file])

    print(f"Preprocessing done")
