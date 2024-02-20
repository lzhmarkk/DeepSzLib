import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, confusion_matrix, fbeta_score
from collections import defaultdict
from tqdm import tqdm

models = ['Shapelet', 'CNN', 'CNNLSTM', 'RNN', 'STGCN', 'MTGNN', 'CNNLSTM', 'DCRNN', 'TapNet', 'Transformer',
          'LinearTransformer', 'FEDFormer', 'CrossFormer', 'SageFormer', 'DSN']
alias = {"CNN": "DenseCNN", "RNN": "SegRNN", "Transformer": "TSD", "CNNLSTM": "CNN-LSTM",
         "DCRNN": "DCRNN-dist", "LinearTransformer": "LTransformer", 'DSN': "DSN"}
fontsize = 20
cmap = plt.colormaps.get_cmap('tab20').colors
colors = {"RNN": cmap[0], "CNNLSTM": cmap[2], "DCRNN": cmap[4], "LinearTransformer": cmap[8], "DSN": cmap[6],
          "CNN": cmap[10], "STGCN": cmap[12], "MTGNN": cmap[14], "Transformer": cmap[16], 'FEDFormer': cmap[18], 'CrossFormer': cmap[1],
          'SageFormer': cmap[3], 'Shapelet': cmap[5], "TapNet": cmap[7]}
run_names = {
    "DCRNN": "baseline-dist",
    "DSN": "tune-ffn"
}


def get_metrics(prob, truth, threshold_value=0.5):
    assert (0 <= prob).all() and (prob <= 1).all()
    metric = {}
    pred = (prob >= threshold_value).astype(int)
    if not np.any(pred):
        return None

    accuracy = accuracy_score(truth, pred)
    precision = precision_score(truth, pred, average='binary')
    recall = recall_score(truth, pred, average='binary')
    f1 = f1_score(truth, pred, average='binary')
    f2 = fbeta_score(truth, pred, beta=2, average='binary')
    auc = roc_auc_score(truth, prob)

    metric['accuracy'] = accuracy
    metric['precision'] = precision
    metric['recall'] = recall
    metric['f1'] = f1
    metric['f2'] = f2
    metric['auc'] = auc
    return metric


if __name__ == '__main__':
    dataset = "TUSZ-Transductive"
    truths = {}
    preds = {}
    for model in models:
        model_path = os.path.join("./saves", dataset, model)
        run_name = run_names[model] if model in run_names else 'baseline'
        # run_name = list(filter(lambda f: 'test' not in f, os.listdir(model_path)))[0]
        data_path = list(filter(lambda f: '.npz' in f, os.listdir(os.path.join(model_path, run_name))))[0]
        data = np.load(os.path.join(model_path, run_name, data_path))
        pred = data['predictions']
        truth = data['targets']

        truths[model] = truth
        preds[model] = pred

    # check
    assert all([np.all(list(truths.values())[0] == truth) for truth in truths.values()])

    # metrics w.r.t threshold
    all_scores = {}
    for model in preds.keys():
        truth = truths[model]
        pred = preds[model]
        metrics_model = defaultdict(list)

        for thres in tqdm(range(0, 100), desc=model):
            thres /= 100
            scores = get_metrics(pred, truth, thres)
            if scores is not None:
                for k, v in scores.items():
                    metrics_model[k].append(v)

        all_scores[model] = metrics_model

    # plot
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    # fig.suptitle("Metrics w.r.t threshold")

    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'auc', 'f1', 'f2']):
        ax = axs[i // 3, i % 3]
        ax.set_title(metric, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        for model in all_scores:
            score = all_scores[model][metric]
            if i == 0:
                label = alias[model] if model in alias else model
                ax.plot(range(len(score)), score, label=label, color=colors[model])
            else:
                ax.plot(range(len(score)), score, color=colors[model])

    """
    # distribution of predictions
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    axs[0].set_title("Distribution of all predicted probability")
    axs[0].hist((pred * 100).astype(int), bins=100)
    correct_mask = ((pred >= data['thres']) == truth)
    wrong_mask = ((pred >= data['thres']) != truth)
    axs[1].set_title("Distribution of all correct predicted probability")
    axs[1].hist((pred[correct_mask] * 100).astype(int), bins=100)
    axs[2].set_title("Distribution of all wrong predicted probability")
    axs[2].hist((pred[wrong_mask] * 100).astype(int), bins=100)
    """

    fig.legend(loc="upper center", fontsize=fontsize, ncols=4, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.75, hspace=0.3)
    plt.savefig("./ExpThreshold.png", dpi=500)
    plt.show()
