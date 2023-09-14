import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, confusion_matrix, fbeta_score
from collections import defaultdict
from tqdm import tqdm


def get_metrics(prob, truth, threshold_value=0.5):
    assert (0 <= prob).all() and (prob <= 1).all()
    metric = {}
    pred = (prob >= threshold_value).astype(int)
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
    for model in os.listdir(os.path.join("./saves", dataset)):
        model_path = os.path.join("./saves", dataset, model)
        run_name = os.listdir(model_path)[0]
        data = np.load(os.path.join(model_path, run_name, "test-results-0.npz"))
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
            for k, v in scores.items():
                metrics_model[k].append(v)

        all_scores[model] = metrics_model

    # plot
    fig, axs = plt.subplots(1, 6, figsize=(50, 6))
    fig.suptitle("Metrics w.r.t threshold")

    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'auc', 'f1', 'f2']):
        ax = axs[i]
        ax.set_title(metric)
        for model in all_scores:
            score = all_scores[model][metric]
            assert len(score) == 100
            ax.plot(range(100), score, label=model)

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

    plt.subplots()
    plt.show()
