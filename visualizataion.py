import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from utils.preprocess import load_edf_data, load_txt_data

dir = "./data/edf_noName_SeizureFile"
file = "20160226_191125-4次发作"
window_step = 30  # seconds
sample_rate = 500
resample_rate = 100

if __name__ == '__main__':
    os.makedirs("./plot", exist_ok=True)

    orig_x, x = load_edf_data(os.path.join(dir, file + ".edf"))
    y = load_txt_data(os.path.join(dir, file + ".txt"), length=x.shape[0], sample_rate=resample_rate)

    # filter and z-norm
    mean = x.mean()
    std = x.std()
    x[x < mean - 3 * std] = 0
    x[x > mean + 3 * std] = 0
    orig_x[orig_x < mean - 3 * std] = 0
    orig_x[orig_x > mean + 3 * std] = 0
    x = (x - mean) / std
    orig_x = (orig_x - mean) / std

    for i in range(x.shape[0] // resample_rate // window_step)[:20]:
        # select a part of data
        _x = x[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        _y = y[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        #
        # if not any(_y):
        #     continue

        # resampled
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].plot(range(window_step * resample_rate), _x[:, feat])
        axs[-1].plot(range(window_step * resample_rate), _y)

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}_resmp.png")
        plt.close()

        # origin
        _x = orig_x[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]
        fig, axs = plt.subplots(_x.shape[1], sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].plot(range(window_step * sample_rate), _x[:, feat])

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}_orig.png")
        plt.close()

        # origin
        from utils.dataloader import compute_FFT

        _x = x[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        _y = y[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        _x = _x.reshape(window_step, resample_rate, 12)
        _x = [compute_FFT(__x.T, n=resample_rate).T for __x in _x]
        _x = np.concatenate(_x, axis=0)
        _y = _y[::2]
        assert len(_x) == len(_y) == window_step * resample_rate // 2

        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].set_xticks(range(0, len(_x), resample_rate // 2))
            axs[feat].plot(range(window_step * resample_rate // 2), _x[:, feat])
        axs[-1].plot(range(window_step * resample_rate // 2), _y)

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}_FFT.png")
        plt.close()
