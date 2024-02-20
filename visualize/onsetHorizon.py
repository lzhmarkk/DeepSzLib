import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

datasets = ["FDUSZ-Transductive", "TUSZ-Transductive"]
models = ["RNN", "STGCN", "CNNLSTM", "DCRNN", "Transformer", "LinearTransformer", "FEDFormer", "CrossFormer", "SageFormer", "DSN"]
alias = {"RNN": "SegRNN", "CNNLSTM": "CNN-LSTM", "DCRNN": "DCRNN-dist", "Transformer": "TSD", "LinearTransformer": "LTransformer",
         "DSN": "DSN"}
markers = {"RNN": '.', "CNNLSTM": "v", "DCRNN": "s", "LinearTransformer": "*", "DSN": "D",
           "CNN": ",", "STGCN": "1", "MTGNN": 'p', "Transformer": '+', 'FEDFormer': 'x', 'CrossFormer': '<', 'SageFormer': '>'}
cmap = plt.colormaps.get_cmap('tab20').colors
colors = {"RNN": cmap[0], "CNNLSTM": cmap[2], "DCRNN": cmap[4], "LinearTransformer": cmap[8], "DSN": cmap[6],
          "CNN": cmap[10], "STGCN": cmap[12], "MTGNN": cmap[14], "Transformer": cmap[16], 'FEDFormer': cmap[18], 'CrossFormer': cmap[1],
          'SageFormer': cmap[3]}
fontsize = 24
k_range = [1, 15]

if __name__ == '__main__':
    # read results
    dr_all, wr_all = {}, {}
    for dataset in datasets:
        dr, wr = defaultdict(list), defaultdict(list)
        for model in models:
            model_path = os.path.join("./saves", dataset, model)
            run_name = list(filter(lambda f: 'onset' in f, os.listdir(model_path)))[0]
            data_path = "test-scores.json"
            with open(os.path.join(model_path, run_name, data_path), 'r') as fp:
                data = json.load(fp)

            for k in range(k_range[0], k_range[1] + 1):
                dr[model].append(data['mean'][f"correct-{k}"])
                wr[model].append(data['mean'][f"wrong-{k}"])

        dr_all[dataset] = dr
        wr_all[dataset] = wr

    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    for j, dataset in enumerate(datasets):
        dr = dr_all[dataset]
        wr = wr_all[dataset]

        # correct rate
        ax = axs[2 * j]
        ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel(f"Horizon\n"
                      f"({['a', 'b', 'c', 'd'][2 * j]})", fontsize=fontsize)
        ax.set_ylabel("Diagnosis Rate", fontsize=fontsize)
        ax.set_yticks([0.0, 0.3, 0.6, 0.7])
        ax.tick_params(labelsize=fontsize)
        for model in dr:
            p = False
            score = dr[model]
            label = alias[model] if model in alias else model
            score = [(i, s) for i, s in enumerate(score) if isinstance(s, float)]
            x = [i for (i, s) in score]
            y = [s for (i, s) in score]
            for i, s in score:
                if not p and s > 0.7:
                    print(model, i)
                    p = True
            if not p:
                print(model, 'inf')
            ax.plot(x, y, marker=markers[model], color=colors[model])
            if j == 0:
                ax.axhline(y=0.6, color='gray', linestyle='--')
                ax.axhline(y=0.7, color='gray', linestyle='--')

        # wrong rate
        ax = axs[2 * j + 1]
        ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel(f"Horizon\n"
                      f"({['a', 'b', 'c', 'd'][2 * j + 1]})", fontsize=fontsize)
        ax.set_ylabel("Wrong Rate", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        for model in wr:
            score = wr[model]
            label = alias[model] if model in alias else model
            score = [(i, s) for i, s in enumerate(score) if isinstance(s, float)]
            x = [i for (i, s) in score]
            y = [s - [0.06, 0.098][j] / 20 * i for (i, s) in score]  # amend the affect of cut off issues when calculating wrong rate
            if j == 0:
                ax.plot(x, y, label=label, marker=markers[model], color=colors[model])
            else:
                ax.plot(x, y, marker=markers[model], color=colors[model])

        print("-" * 30)

    fig.legend(loc="upper center", fontsize=fontsize, ncols=5, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(wspace=10, top=0.75)
    plt.savefig("./ExpHorizon.png", dpi=500)
    plt.show()
