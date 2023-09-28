import matplotlib.pyplot as plt
from collections import OrderedDict

results = {
    "FDUSZ": {
        "w/o SE": [0.533, 0.549, 0.866],
        "w/o RAN": [0.616, 0.611, 0.914],
        "w/o RNN": [0.621, 0.606, 0.910],
        "w/o ESM": [0.578, 0.584, 0.899],
        "w/o ACL": [0.614, 0.616, 0.907],
        "w/o DCL": [0.600, 0.606, 0.902],
        "full DSN": [0.633, 0.620, 0.915]
    },
    "TUSZ": {
        "w/o SE": [0.654, 0.645, 0.947],
        "w/o RAN": [0.719, 0.696, 0.966],
        "w/o RNN": [0.715, 0.683, 0.964],
        "w/o ESM": [0.699, 0.682, 0.961],
        "w/o ACL": [0.720, 0.709, 0.964],
        "w/o DCL": [0.717, 0.695, 0.962],
        "full DSN": [0.739, 0.727, 0.968]
    },
    "CHBMIT": {
        "w/o SE": [0.381, 0.382, 0.842],
        "w/o RAN": [0.583, 0.551, 0.947],
        "w/o RNN": [0.558, 0.507, 0.944],
        "w/o ESM": [0.510, 0.423, 0.935],
        "w/o ACL": [0.585, 0.519, 0.944],
        "w/o DCL": [0.552, 0.524, 0.941],
        "full DSN": [0.611, 0.570, 0.949]
    }
}
metrics = ['F1', 'F2', 'AUC']
width = .125
gap = 1.15
colors = plt.colormaps['Set3'].colors
hatches = ['//', 'x', 'x', 'x', '\\\\', '\\\\', '-']  # '++', '*', 'O', 'o', '.', '/'
fontsize = 24

if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots(sharey=False)

    datasets = ["FDUSZ", "TUSZ", "CHBMIT"]
    for i, dataset in enumerate(datasets):
        result = OrderedDict(results[dataset])
        ax = axs[i]
        ax.set_xticks(range(3), metrics)
        ax.tick_params(labelsize=fontsize)

        # F1/F2
        ax.set_ylabel('F1 / F2', fontsize=fontsize)
        ax.set_xlabel(f"({['a', 'b', 'c'][i]}) {dataset}", fontsize=fontsize)
        if dataset == 'FDUSZ':
            ax.set_ylim(0.5, 0.65)
        elif dataset == 'TUSZ':
            ax.set_ylim(0.6, 0.75)
        else:
            ax.set_ylim(0.3, 0.7)

        for j, metric in enumerate(metrics[:2]):
            for k, variate in enumerate(result):
                x = j + (k - len(result) / 2) * width
                y = result[variate][j]
                if i == 0 and j == 0:
                    ax.bar(x, y, width * 0.9, color=colors[k + 1], hatch=hatches[k], label=variate)
                else:
                    ax.bar(x, y, width * 0.9, color=colors[k + 1], hatch=hatches[k])

        # AUC
        ax = ax.twinx()
        ax.set_ylabel(r'AUC', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        if dataset == 'FDUSZ':
            ax.set_ylim(0.85, 0.95)
            ax.set_yticks([0.8, 0.9])
        elif dataset == 'TUSZ':
            ax.set_ylim(0.9, 1)
            ax.set_yticks([0.9, 1.0])
        else:
            ax.set_ylim(0.8, 1)
            ax.set_yticks([0.8, 0.9, 1.0])
        for j, metric in enumerate(metrics[2:]):
            for k, variate in enumerate(result):
                x = j + 2 + (k - len(result) / 2) * width
                y = result[variate][j + 2]
                ax.bar(x, y, width * 0.9, color=colors[k + 1], hatch=hatches[k])

    fig.legend(loc="upper center", fontsize=fontsize, ncols=7, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.savefig("./ExpAblation.png", dpi=500)
    plt.show()
