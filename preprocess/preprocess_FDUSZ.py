import os
import mne
import argparse
import numpy as np
from tqdm import tqdm
from scipy.signal import resample
from process import process
from multiprocessing import Pool

origin_dir = f"./data/original_dataset/FDUSZ"
dest_dir = f"./data/FDUSZ"
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
    truth = np.zeros([length], dtype=int)

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


def process_patient_file(args):
    f, patient_dir, sample_rate, user_id = args
    _, x = load_edf_data(os.path.join(patient_dir, f + ".edf"), sample_rate)
    y = load_truth_data(os.path.join(patient_dir, f + ".txt"), length=x.shape[0], sample_rate=sample_rate)
    return (user_id, x, y)

def process_control_file(args):
    f, patient_dir, sample_rate, user_id = args
    _, x = load_edf_data(os.path.join(patient_dir, f + ".edf"), sample_rate)
    y = load_truth_data(None, length=x.shape[0], sample_rate=sample_rate)
    return (user_id, x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--setting", type=str, choices=["Inductive", "Transductive"], required=True)
    parser.add_argument("--split", type=str, default="7/1/2")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--patch_len", type=int, default=1)
    args = parser.parse_args()

    sample_rate = args.sample_rate
    setting = args.setting
    split = args.split
    window = args.window
    horizon = args.horizon
    stride = args.stride
    patch_len = args.patch_len
    ratio = [float(r) for r in str(split).split('/')]
    ratio = [r / sum(ratio) for r in ratio]

    # load data
    all_u, all_x, all_y = [], [], []
    patient_dir = os.path.join(origin_dir, 'edf_noName_SeizureFile')
    patient_files = sorted(list(set([_[:-4] for _ in os.listdir(patient_dir)])))
    tasks = [(f, patient_dir, sample_rate, i) for i, f in enumerate(patient_files)]
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_patient_file, tasks), total=len(tasks),
                            desc="Loading patients in parallel"))
    for user_id, x, y in results:
        all_u.append(user_id)
        all_x.append(x)
        all_y.append(y)

    control_dir = os.path.join(origin_dir, 'control')
    control_files = sorted(list(set([_[:-4] for _ in os.listdir(control_dir)])))
    tasks = [(f, control_dir, sample_rate, i + len(all_u)) for i, f in enumerate(control_files)]
    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(process_control_file, tasks), total=len(tasks),
                            desc="Loading control in parallel"))
        for user_id, x, y in results:
            all_u.append(user_id)
            all_x.append(x)
            all_y.append(y)

    dest_dir = dest_dir + '-' + setting
    process(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, setting, ratio, dest_dir, split, channels, n_sample_per_file)
