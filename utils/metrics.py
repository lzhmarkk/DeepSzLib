import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve


def get_metrics(pred, truth,threshold_value=0.5):
    pred = pred.copy()
    truth = truth.copy()

    metric = {}
    pred[pred >= threshold_value] = 1
    pred[pred < threshold_value] = 0
    accuracy = accuracy_score(truth, pred)
    precision = precision_score(truth, pred, average='binary')
    recall = recall_score(truth, pred, average='binary')
    f1 = f1_score(truth, pred, average='binary')
    auc = roc_auc_score(truth, pred)

    metric['accuracy'] = accuracy
    metric['precision'] = precision
    metric['recall'] = recall
    metric['f1'] = f1
    metric['auc'] = auc
    return metric


def thresh_max_f1(y_true, y_prob):
    """
    Find the best threshold based on precision-recall curve to maximize F1-score.
    Binary classification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filtered = []
    f_score = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            f_score.append(curr_f1)
            thresh_filtered.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(f_score))
    best_thresh = thresh_filtered[ix]
    return best_thresh
