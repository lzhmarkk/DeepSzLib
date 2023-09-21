import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

datasets = ["TUSZ-Transductive", "FDUSZ-Transductive"]
models = ["RNN", "CNNLSTM", "DCRNN", "LinearTransformer", "DualGraph"]
alias = {"RNN": "SegRNN", "CNNLSTM": "CNN-LSTM", "DCRNN": "DCRNN-dist", "LinearTransformer": "LTransformer",
         "DualGraph": "ours"}
fontsize = 15
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
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for j, dataset in enumerate(datasets):
        dr = dr_all[dataset]
        wr = wr_all[dataset]

        # correct rate
        ax = axs[2 * j]
        # ax.set_title("Diagnosis Rate", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        for model in dr:
            score = dr[model]
            label = alias[model] if model in alias else model
            score = [(i, s) for i, s in enumerate(score) if isinstance(s, float)]
            x = [i for (i, s) in score]
            y = [s for (i, s) in score]
            ax.plot(x, y)

        # wrong rate
        ax = axs[2 * j + 1]
        # ax.set_title("Wrong Rate", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        for model in wr:
            score = wr[model]
            label = alias[model] if model in alias else model
            score = [(i, s) for i, s in enumerate(score) if isinstance(s, float)]
            x = [i for (i, s) in score]
            y = [s - [0.06, 0.098][j] / 20 * i for (i, s) in score]  # amend the affect of cut off issues when calculating wrong rate
            if j == 0:
                ax.plot(x, y, label=label)
            else:
                ax.plot(x, y)

    fig.legend(loc="upper center", fontsize=fontsize, ncols=7, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig("./ExpHorizon.png", dpi=500)
    plt.show()
