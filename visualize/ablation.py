import matplotlib.pyplot as plt
from collections import OrderedDict

results = {
    "FDUSZ": {
        "w/o SE": [0.543, 0.569, 0.876],
        "w/o DyAN": [0.609, 0.606, 0.912],
        # "w/o RNN": [0.621, 0.606, 0.910],
        "w/o ESM": [0.581, 0.590, 0.899],
        "w/o ACL": [0.620, 0.623, 0.910],
        "w/o DCL": [0.608, 0.610, 0.906],
        "full DSN": [0.632, 0.641, 0.915]
    },
    "TUSZ": {
        "w/o SE": [0.654, 0.645, 0.947],
        "w/o DyAN": [0.719, 0.696, 0.966],
        # "w/o RNN": [0.715, 0.683, 0.964],
        "w/o ESM": [0.699, 0.682, 0.961],
        "w/o ACL": [0.720, 0.709, 0.964],
        "w/o DCL": [0.717, 0.695, 0.962],
        "full DSN": [0.739, 0.727, 0.968]
    },
    "CHBMIT": {
        "w/o SE": [0.381, 0.382, 0.842],
        "w/o DyAN": [0.583, 0.551, 0.947],
        # "w/o RNN": [0.558, 0.507, 0.944],
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
hatches = ['//', 'x', 'x', 'x', '\\\\', '\\\\', '++']  # '++', '*', 'O', 'o', '.', '/'
fontsize = 20

if __name__ == '__main__':
    # FDUSZ & TUSZ
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots(sharey=False)

    datasets = ["FDUSZ", "TUSZ"]
    for i, dataset in enumerate(datasets):
        result = OrderedDict(results[dataset])
        ax = axs[i]
        ax.set_xticks(range(3), metrics)
        ax.tick_params(labelsize=fontsize)

        # F1/F2
        ax.set_ylabel('F1 / F2', fontsize=fontsize)
        ax.set_xlabel(f"({['a', 'b', 'c'][i]}) {dataset}", fontsize=fontsize + 4)
        if dataset == 'FDUSZ':
            ax.set_ylim(0.5, 0.65)
            ax.set_yticks([0.5, 0.6])
        elif dataset == 'TUSZ':
            ax.set_ylim(0.6, 0.75)
            ax.set_yticks([0.6, 0.7])
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
            ax.set_ylim(0.85, 0.93)
            ax.set_yticks([0.85, 0.9])
        elif dataset == 'TUSZ':
            ax.set_ylim(0.94, 0.97)
            ax.set_yticks([0.94, 0.95, 0.96])
        else:
            ax.set_ylim(0.8, 0.97)
            ax.set_yticks([0.8, 0.9])
        for j, metric in enumerate(metrics[2:]):
            for k, variate in enumerate(result):
                x = j + 2 + (k - len(result) / 2) * width
                y = result[variate][j + 2]
                ax.bar(x, y, width * 0.9, color=colors[k + 1], hatch=hatches[k])

    fig.legend(loc="upper center", fontsize=fontsize, ncols=4, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.78, wspace=0.5)
    plt.savefig("./ExpAblation1.png", dpi=500)
    plt.show()

    # CHBMIT
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 10)
    axs = plt.subplot(gs[0, 3:7])
    datasets = ["CHBMIT"]
    for i, dataset in enumerate(datasets):
        result = OrderedDict(results[dataset])
        ax = axs
        ax.set_xticks(range(3), metrics)
        ax.tick_params(labelsize=fontsize)

        # F1/F2
        ax.set_ylabel('F1 / F2', fontsize=fontsize)
        ax.set_xlabel(f"({['a', 'b', 'c'][i]}) {dataset}", fontsize=fontsize + 4)
        if dataset == 'FDUSZ':
            ax.set_ylim(0.5, 0.65)
            ax.set_yticks([0.5, 0.6])
        elif dataset == 'TUSZ':
            ax.set_ylim(0.6, 0.75)
            ax.set_yticks([0.6, 0.7])
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
        ax.set_ylim(0.8, 0.97)
        ax.set_yticks([0.8, 0.9])
        for j, metric in enumerate(metrics[2:]):
            for k, variate in enumerate(result):
                x = j + 2 + (k - len(result) / 2) * width
                y = result[variate][j + 2]
                ax.bar(x, y, width * 0.9, color=colors[k + 1], hatch=hatches[k])

    fig.legend(loc="upper center", fontsize=fontsize, ncols=4, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.78, wspace=0.5)
    plt.savefig("./ExpAblation2.png", dpi=500)
    plt.show()