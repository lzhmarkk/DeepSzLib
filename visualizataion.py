import os
import numpy as np
from utils.dataloader import load_data, filter
import matplotlib.pyplot as plt

dir = "./data/FDUSZ"
file = "20160226_191125-4次发作"
HEAD = 1000  # drop head
sigma = 3
downSampling = 10  # 50hz is a good balance
hz = int(500 / downSampling)

if __name__ == '__main__':
    os.makedirs("./plot", exist_ok=True)

    # load files from disk
    x, y = [], []
    files = os.listdir(dir)
    n_users = len(files)
    for f in files:
        data = np.load(os.path.join(dir, f))
        x.append(data['x'])  # (T, C)
        y.append(data['y'])  # (T)
        sample_rate = data['sr'].item()

    # abnormal values
    print(f"\nabnormal records:")
    data_filtered = []
    n_abnormal = 0
    for f in range(data.shape[0]):
        _data_filtered, _n_abnormal = filter(data[f], sigma)
        print(f"feat {f}: {_n_abnormal}/{data[f].size}, {_n_abnormal / data[f].size}")
        data_filtered.append(_data_filtered)
        n_abnormal += _n_abnormal
    print(f"all: {n_abnormal}/{data.size}, {n_abnormal / data.size}\n")
    data = np.stack(data_filtered, axis=0)

    for i in range(data.shape[1] // hz // 300):
        # select a part of data
        _data = data[:, 300 * i * hz:300 * (i + 1) * hz]
        _truth = truth[300 * i * hz:300 * (i + 1) * hz]

        fig, axs = plt.subplots(_data.shape[0] + 1, sharex='all', figsize=(_data.shape[1] / hz, _data.shape[0] * 2))
        x = range(_data.shape[1])
        for feat in range(_data.shape[0]):
            ax = axs[feat]
            y = _data[feat]
            ax.plot(x, y)
        axs[-1].plot(x, _truth)

        plt.tight_layout()
        plt.savefig(f"./plot/visualization_{i}_{'sei' if any(_truth == 1) else ''}_{downSampling}.png")
        plt.close()
