import os
import mne
import h5py
import numpy as np
from tqdm import tqdm
from scipy.signal import resample

dir = f"./data/"
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'T3', 'T4', 'EKG', 'EMG']
sample_rate = 500
resample_rate = 100


def load_edf_data(edf_path):
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
    dataset_path = os.path.join(dir, 'FDUSZ')
    os.makedirs(dataset_path, exist_ok=True)

    patient_dir = os.path.join(dir, 'edf_noName_SeizureFile')
    control_dir = os.path.join(dir, 'control')
    patient_files = os.listdir(patient_dir)
    control_files = os.listdir(control_dir)
    patient_files = sorted(list(set([_[:-4] for _ in patient_files])))
    control_files = sorted(list(set([_[:-4] for _ in control_files])))
    n_users = len(patient_files) + len(control_files)

    user_id = 0
    for f in tqdm(patient_files, desc="Loading patient"):
        _, x = load_edf_data(os.path.join(patient_dir, f + ".edf"))
        y = load_txt_data(os.path.join(patient_dir, f + ".txt"), length=x.shape[0], sample_rate=resample_rate)

        user_path = os.path.join(dataset_path, f"{user_id}.h5")
        with h5py.File(user_path, "w") as hf:
            hf.create_dataset("x", data=x)
            hf.create_dataset("y", data=y)
            hf.create_dataset("sr", data=resample_rate)
        user_id += 1

    for f in tqdm(control_files, desc="Loading control"):
        _, x = load_edf_data(os.path.join(control_dir, f + ".edf"))
        y = load_txt_data(None, length=x.shape[0], sample_rate=resample_rate)

        user_path = os.path.join(dataset_path, f"{user_id}.h5")
        with h5py.File(user_path, "w") as hf:
            hf.create_dataset("x", data=x)
            hf.create_dataset("y", data=y)
            hf.create_dataset("sr", data=resample_rate)
        user_id += 1

    print(f"Save {user_id} users")
