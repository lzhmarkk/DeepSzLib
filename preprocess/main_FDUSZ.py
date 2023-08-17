import os
import mne
import json
import numpy as np
from tqdm import tqdm
from scipy.signal import resample
from process import process

origin_dir = f"./data/original_dataset/FDUSZ"
dest_dir = f"./data/FDUSZ/"
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'T3', 'T4', 'EKG', 'EMG']
n_sample_per_file = 1000
np.random.seed(0)


def load_edf_data(edf_path, sample_rate):
    data = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
    orig_sample_rate = int(data.info['sfreq'])

    data_channels = []
    for c in channels:
        matched_channels = sorted(list(filter(lambda e: c in e, data.ch_names)))
        matched_channel = matched_channels[0]

        data_channels.append(data[matched_channel][0])

    data = np.concatenate(data_channels, axis=0).T  # (T, C)
    assert data.shape[1] == len(channels)

    resample_data = resample(data, num=data.shape[0] // orig_sample_rate * sample_rate, axis=0)
    return data, resample_data


def load_truth_data(txt_path, length, sample_rate):
    truth = np.zeros([length], dtype=float)

    if txt_path is None:
        return truth

    timestamps = []
    with open(txt_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    s = False
    for i, l in enumerate(lines):
        words = str(l).split('\t')
        if not s:
            if '开始' in words[0]:
                timestamps.append(words[1])
                s = not s
        else:
            if '结束' in words[0]:
                timestamps.append(words[1])
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

    # load data
    user_id = 0
    all_x, all_y = [], []
    patient_dir = os.path.join(origin_dir, 'edf_noName_SeizureFile')
    patient_files = sorted(list(set([_[:-4] for _ in os.listdir(patient_dir)])))
    for f in tqdm(patient_files, desc="Loading patient"):
        _, x = load_edf_data(os.path.join(patient_dir, f + ".edf"), sample_rate)
        y = load_truth_data(os.path.join(patient_dir, f + ".txt"), length=x.shape[0], sample_rate=sample_rate)
        all_x.append(x)
        all_y.append(y)
        user_id += 1

    control_dir = os.path.join(origin_dir, 'control')
    control_files = sorted(list(set([_[:-4] for _ in os.listdir(control_dir)])))
    for f in tqdm(control_files, desc="Loading control"):
        _, x = load_edf_data(os.path.join(control_dir, f + ".edf"), sample_rate)
        y = load_truth_data(None, length=x.shape[0], sample_rate=sample_rate)
        all_x.append(x)
        all_y.append(y)
        user_id += 1

    process(all_x, all_y, sample_rate, window, horizon, stride, seg, mode, ratio, dest_dir, split, channels, n_sample_per_file)
