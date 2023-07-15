import os
import numpy as np
import matplotlib.pyplot as plt
from preprocess.main import load_edf_data, load_txt_data
from preprocess.utils import compute_FFT

dir = "./data/edf_noName_SeizureFile"
file = "20160226_191125-4次发作"
window_step = 30  # seconds
sample_rate = 500
resample_rate = 100
seg = 1

if __name__ == '__main__':
    os.makedirs("./plot", exist_ok=True)

    orig_x, x = load_edf_data(os.path.join(dir, file + ".edf"), sample_rate, resample_rate)
    y = load_txt_data(os.path.join(dir, file + ".txt"), length=x.shape[0], sample_rate=resample_rate)

    for i in range(x.shape[0] // resample_rate // window_step):
        # select a part of data
        _x = x[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        _y = y[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]

        if not any(_y):
            continue

        # resampled
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].plot(range(window_step * resample_rate), _x[:, feat])
        axs[-1].plot(range(window_step * resample_rate), _y)

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
        x_fft = x[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        y_fft = y[i * window_step * resample_rate: (i + 1) * window_step * resample_rate]
        x_fft = [x_fft[i * (seg * resample_rate):(i + 1) * (seg * resample_rate), :]
                 for i in range(len(x_fft) // (seg * resample_rate))]
        x_fft_noCat = [compute_FFT(_seg.T, n=seg * resample_rate).T for _seg in x_fft]
        x_fft = np.concatenate(x_fft_noCat, axis=0)
        y_fft = y_fft[::2]
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1]):
            axs[feat].plot(x_fft[:, feat])
        axs[-1].plot(y_fft)
        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(y_fft == 1) else ''}_FFT.png")
        plt.close()
