import os
import mne
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import resample
from process import process_TUSZ

origin_dir = f"./data/original_dataset/TUSZ"
dest_dir = f"./data/TUSZ/"
channels = ["EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4", "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7",
            "EEG F8", "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ", "EEG PZ", ]
n_sample_per_file = 1000
np.random.seed(0)


def load_edf_data(edf_path, sample_rate):
    data = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
    orig_smp_rate = int(data.info['sfreq'])

    data_channels = []
    for c in channels:
        matched_channels = sorted(list(filter(lambda e: c in e, data.ch_names)))
        matched_channel = matched_channels[0]

        data_channels.append(data[matched_channel][0])

    data = np.concatenate(data_channels, axis=0).T  # (T, C)
    assert data.shape[1] == len(channels)

    resample_data = resample(data, num=data.shape[0] // orig_smp_rate * sample_rate, axis=0)
    return data, resample_data


def load_truth_data(csv_path, length, sample_rate):
    truth = np.zeros([length], dtype=float)

    df = pd.read_csv(csv_path, header=0, comment='#')
    df = df[df['label'].str.endswith('z')]
    if len(df) > 0:
        s_time = df['start_time'].min()
        e_time = df['stop_time'].max()
        s_time = int(s_time * sample_rate)
        e_time = int(e_time * sample_rate)
        truth[s_time:e_time] = 1

    return truth


if __name__ == '__main__':
    with open("./preprocess/config.json", 'r') as fp:
        config = json.load(fp)
        sample_rate = config["resample_rate"]
        preprocess = config["preprocess"]
        mode = config["mode"]
        split = config["split"]
        ratio = [float(r) for r in str(split).split('/')]
        ratio = [r / sum(ratio) for r in ratio]
        window = config["window"]
        horizon = config["horizon"]
        stride = config["stride"]
        seg = config["seg"]

    # load paths
    train_files, val_files, test_files = [], [], []
    for u in os.listdir(os.path.join(origin_dir, 'edf', 'train')):
        for dir in os.listdir(os.path.join(origin_dir, 'edf', 'train', u)):
            for subdir in os.listdir(os.path.join(origin_dir, 'edf', 'train', u, dir)):
                cur_dir = os.path.join(origin_dir, 'edf', 'train', u, dir, subdir)
                for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                    train_files.append((cur_dir, f))
    for u in os.listdir(os.path.join(origin_dir, 'edf', 'dev')):
        for dir in os.listdir(os.path.join(origin_dir, 'edf', 'dev', u)):
            for subdir in os.listdir(os.path.join(origin_dir, 'edf', 'dev', u, dir)):
                cur_dir = os.path.join(origin_dir, 'edf', 'dev', u, dir, subdir)
                for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                    val_files.append((cur_dir, f))
    for u in os.listdir(os.path.join(origin_dir, 'edf', 'eval')):
        for dir in os.listdir(os.path.join(origin_dir, 'edf', 'eval', u)):
            for subdir in os.listdir(os.path.join(origin_dir, 'edf', 'eval', u, dir)):
                cur_dir = os.path.join(origin_dir, 'edf', 'eval', u, dir, subdir)
                for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                    test_files.append((cur_dir, f))

    # load data
    for mode in ['train', 'val', 'test']:
        all_x, all_y = [], []
        if mode == 'train':
            files = train_files
        elif mode == 'val':
            files = val_files
        else:
            files = test_files

        for cur_dir, f in tqdm(files):
            _, x = load_edf_data(os.path.join(cur_dir, f + ".edf"), sample_rate)
            y = load_truth_data(os.path.join(cur_dir, f + ".csv"), length=x.shape[0], sample_rate=sample_rate)
            all_x.append(x)
            all_y.append(y)

        process_TUSZ(all_x, all_y, sample_rate, window, horizon, stride, seg, mode, dest_dir, n_sample_per_file)
