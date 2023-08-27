import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from preprocess.utils import compute_FFT
from scipy.signal import resample
import scipy

dir = "../data/original_dataset/FDUSZ/edf_noName_SeizureFile"
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'T3', 'T4', 'EKG', 'EMG']
file = "20160226_191125-4次发作"
window_step = 30  # seconds
sample_rate = 100
seg = 1


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
    os.makedirs("../plot", exist_ok=True)

    orig_x, x = load_edf_data(os.path.join(dir, file + ".edf"), sample_rate)
    y = load_truth_data(os.path.join(dir, file + ".txt"), length=x.shape[0], sample_rate=sample_rate)

    for i in range(x.shape[0] // sample_rate // window_step):
        # select a part of data
        _x = x[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]
        _y = y[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]

        if not any(_y):
            continue

        # resampled
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].plot(range(window_step * sample_rate), _x[:, feat])
        axs[-1].plot(range(window_step * sample_rate), _y)

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}_resmp.png")
        plt.close()

        # origin
        # _x = orig_x[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]
        # fig, axs = plt.subplots(_x.shape[1], sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        # for feat in range(0, _x.shape[1]):
        #     axs[feat].plot(range(window_step * sample_rate), _x[:, feat])
        #
        # plt.tight_layout()
        # plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}_orig.png")
        # plt.close()

        # fft
        x_fft = x[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]
        y_fft = y[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]
        x_fft = [x_fft[i * (seg * sample_rate):(i + 1) * (seg * sample_rate), :]
                 for i in range(len(x_fft) // (seg * sample_rate))]
        x_fft_noCat = [compute_FFT(_seg.T, n=seg * sample_rate).T for _seg in x_fft]
        x_fft = np.concatenate(x_fft_noCat, axis=0)
        y_fft = y_fft[::2]
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].plot(resample(x_fft[:, feat], num=window_step*sample_rate))
        axs[-1].plot(y_fft)
        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(y_fft == 1) else ''}_FFT.png")
        plt.close()
