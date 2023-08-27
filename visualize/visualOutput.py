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
    auc = roc_auc_score(truth, pred)
    mtx = confusion_matrix(truth, pred)

    metric['accuracy'] = accuracy
    metric['precision'] = precision
    metric['recall'] = recall
    metric['f1'] = f1
    metric['f2'] = fbeta_score(truth, pred, beta=2)
    metric['f3'] = fbeta_score(truth, pred, beta=3)
    metric['f4'] = fbeta_score(truth, pred, beta=4)
    metric['f6'] = fbeta_score(truth, pred, beta=6)
    metric['f8'] = fbeta_score(truth, pred, beta=9)
    metric['auc'] = auc
    metric['confusion_matrix'] = mtx
    return metric


if __name__ == '__main__':
    data = np.load("./saves/FDUSZ/DualGraph/sigmoid/test-results-0.npz")
    pred = data['predictions']
    truth = data['targets']

    # metrics w.r.t threshold
    all_metrics = defaultdict(list)
    fig, axs = plt.subplots(2, 5, figsize=(30, 10))
    fig.suptitle("Metrics w.r.t threshold")
    for thres in tqdm(range(0, 100)):
        thres /= 100
        metric = get_metrics(pred, truth, thres)
        for k, v in metric.items():
            all_metrics[k].append(v)
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'auc', 'f1']):
        values = all_metrics[metric]
        ax = axs[0, i]
        ax.set_title(metric)
        ax.plot(range(100), values)
    for i, metric in enumerate(['f2', 'f3', 'f4', 'f6', 'f8']):
        values = all_metrics[metric]
        ax = axs[1, i]
        ax.set_title(metric)
        ax.plot(range(100), values)

    # confusion matrix
    # [[33061  1719]
    #  [ 1338  2422]]
    confusion_mtx = get_metrics(pred, truth, data['thres'])['confusion_matrix']
    print(confusion_mtx)

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

    plt.show()
