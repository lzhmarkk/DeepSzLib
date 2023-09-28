import matplotlib.pyplot as plt

models = ["SegRNN", "DCRNN", "TSD", "CrossFormer", "DSN"]
window = [1, 5, 10, 15, 30]
markers = {"SegRNN": '.', "DCRNN": "s", "DSN": "D", "TSD": '+', 'CrossFormer': '<'}
cmap = plt.colormaps.get_cmap('tab20').colors
colors = {"SegRNN": cmap[0], "DCRNN": cmap[4], "DSN": cmap[6], "TSD": cmap[16], 'CrossFormer': cmap[1]}

results = {
    "SegRNN": [[0.361, 0.407, 0.784, 0.478, 0.502, 0.876], [0.439, 0.463, 0.828, 0.579, 0.590, 0.917],
               [0.472, 0.524, 0.853, 0.612, 0.613, 0.926], [0.497, 0.522, 0.860, 0.613, 0.607, 0.932],
               [0.531, 0.557, 0.872, 0.640, 0.637, 0.940]],
    "DCRNN": [[0.287, 0.366, 0.713, 0.351, 0.407, 0.813], [0.412, 0.442, 0.801, 0.470, 0.497, 0.870],
              [0.485, 0.509, 0.844, 0.531, 0.537, 0.896], [0.485, 0.497, 0.847, 0.558, 0.577, 0.907],
              [0.555, 0.560, 0.877, 0.572, 0.596, 0.924]],
    "TSD": [[0.374, 0.412, 0.785, 0.456, 0.472, 0.865], [0.474, 0.488, 0.843, 0.573, 0.575, 0.914],
            [0.498, 0.503, 0.853, 0.545, 0.546, 0.907], [0.507, 0.511, 0.854, 0.520, 0.536, 0.903],
            [0.547, 0.555, 0.882, 0.500, 0.536, 0.909]],
    "CrossFormer": [[0.391, 0.429, 0.802, 0.468, 0.481, 0.870], [0.510, 0.513, 0.853, 0.610, 0.583, 0.928],
                    [0.540, 0.539, 0.878, 0.657, 0.639, 0.941], [0.571, 0.570, 0.886, 0.687, 0.659, 0.954],
                    [0.578, 0.574, 0.898, 0.713, 0.722, 0.964]],
    "DSN": [[0.397, 0.424, 0.809, 0.574, 0.576, 0.913], [0.525, 0.541, 0.873, 0.651, 0.668, 0.947],
            [0.551, 0.573, 0.886, 0.670, 0.677, 0.948], [0.572, 0.579, 0.893, 0.701, 0.706, 0.958],
            [0.633, 0.620, 0.915, 0.739, 0.727, 0.968]]}

fontsize = 15

if __name__ == '__main__':
    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # F1, FDUSZ
    axs[0].set_xlabel("Window Length (s)\n" + r"(a) Detection $F_1$ on FDUSZ", fontsize=fontsize)
    axs[0].set_ylabel(r"$F_1$", fontsize=fontsize)
    for m in models:
        data = [e[0] for e in results[m]]
        axs[0].plot(window, data, label=m, marker=markers[m], color=colors[m])

    # AUC, FDUSZ
    axs[1].set_xlabel("Window Length (s)\n" + r"(a) Detection $AUC$ on FDUSZ", fontsize=fontsize)
    axs[1].set_ylabel(r"$AUC$", fontsize=fontsize)
    for m in models:
        data = [e[2] for e in results[m]]
        axs[1].plot(window, data, marker=markers[m], color=colors[m])

    # F1, TUSZ
    axs[2].set_xlabel("Window Length (s)\n" + r"(a) Detection $F_1$ on TUSZ", fontsize=fontsize)
    axs[2].set_ylabel(r"$F_1$", fontsize=fontsize)
    for m in models:
        data = [e[3] for e in results[m]]
        axs[2].plot(window, data, marker=markers[m], color=colors[m])

    # AUC, TUSZ
    axs[3].set_xlabel("Window Length (s)\n" + r"(a) Detection $AUC$ on TUSZ", fontsize=fontsize)
    axs[3].set_ylabel(r"$AUC$", fontsize=fontsize)
    for m in models:
        data = [e[5] for e in results[m]]
        axs[3].plot(window, data, marker=markers[m], color=colors[m])

    fig.legend(loc="upper center", fontsize=fontsize, ncols=6, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.78)
    plt.savefig("./ExpWindow.png", dpi=500)
    plt.show()
