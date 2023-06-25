import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from utils.preprocess import load_edf_data, load_txt_data

dir = "./data/edf_noName_SeizureFile"
file = "20160222_212106-3次发作"
window_step = 12  # seconds
sample_rate = 500
resample_rate = 200

if __name__ == '__main__':
    os.makedirs("./plot", exist_ok=True)

    x = load_edf_data(os.path.join(dir, file + ".edf"))
    y = load_txt_data(os.path.join(dir, file + ".txt"), length=x.shape[0])

    # filter and z-norm
    mean = x.mean()
    std = x.std()
    x[x < mean - 3 * std] = 0
    x[x > mean + 3 * std] = 0
    x = (x - x.mean()) / x.std()

    for i in range(x.shape[0] // sample_rate // window_step):
        # select a part of data
        _x = x[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]
        _y = y[i * window_step * sample_rate: (i + 1) * window_step * sample_rate]

        # original
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1], 2):
            axs[feat].plot(range(window_step * sample_rate), _x[:, feat])
        axs[-1].plot(range(window_step * sample_rate), _y)

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}.png")
        plt.close()

        # resample
        fig, axs = plt.subplots(_x.shape[1] + 1, sharex='all', figsize=(window_step * 2, _x.shape[1] * 2))
        for feat in range(0, _x.shape[1], 2):
            resample_x = resample(_x[:, feat], num=window_step * resample_rate, axis=0)
            axs[feat].plot(range(window_step * resample_rate), resample_x)

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_y == 1) else ''}_rsmpl.png")
        plt.close()
