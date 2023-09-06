import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, fbeta_score
from collections import defaultdict


def iou_score(truth, pred):
    intersection = np.logical_and(pred, truth)
    union = np.logical_or(pred, truth)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def dice_coefficient(truth, pred):
    intersection = np.logical_and(pred, truth)
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(truth))
    return dice_coefficient


def consistency_score(truth, pred):
    # evaluate the difference of successive time stamps
    assert truth.ndim == 2 and pred.ndim == 2
    return np.sum(pred[:, 1:] == pred[:, :-1]) / np.size(pred[:, 1:])


def delay_metrics(truth, pred):
    metric = {}
    pred_horizons = defaultdict(list)
    for _truth, _pred in zip(truth, pred):
        if _truth.any():
            onset = _truth.nonzero()[0].min()
            for delta in range(truth.shape[1] - onset):
                _truth_delta = _truth[onset:onset + delta + 1].any()
                _pred_delta = _pred[onset:onset + delta + 1].any()
                assert _truth_delta
                pred_horizons[delta + 1].append(_pred_delta)

    for delta in range(truth.shape[1]):
        pred = pred_horizons[delta]
        truth = np.ones_like(pred)
        recall = recall_score(truth, pred, average='binary')
        metric[f'recall-{delta}'] = recall

    return metric


def get_metrics(prob, truth, threshold_value=0.5):
    assert (0 <= prob).all() and (prob <= 1).all()
    assert prob.shape == truth.shape

    metric = {}
    pred = (prob >= threshold_value).astype(int)

    if prob.ndim == 1 and truth.ndim == 1:
        # classification
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

    elif prob.ndim == 2 and truth.ndim == 2:
        truth_flat = truth.flatten()
        pred_flat = pred.flatten()
        accuracy = accuracy_score(truth_flat, pred_flat)
        precision = precision_score(truth_flat, pred_flat, average='binary')
        recall = recall_score(truth_flat, pred_flat, average='binary')
        f1 = f1_score(truth_flat, pred_flat, average='binary')
        f2 = fbeta_score(truth_flat, pred_flat, beta=2, average='binary')
        auc = roc_auc_score(truth_flat, pred_flat)
        iou = iou_score(truth_flat, pred_flat)
        dice = dice_coefficient(truth_flat, pred_flat)
        consistency = consistency_score(truth, pred)
        delay_metric = delay_metrics(truth, pred)

        metric['accuracy'] = accuracy
        metric['precision'] = precision
        metric['recall'] = recall
        metric['f1'] = f1
        metric['f2'] = f2
        metric['auc'] = auc
        metric['iou'] = iou
        metric['dice'] = dice
        metric['consist'] = consistency
        metric.update(delay_metric)

    return metric


def thresh_max_f1(y_true, y_prob):
    """
    Find the best threshold based on precision-recall curve to maximize F1-score.
    Binary classification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError
    assert (0 <= y_prob).all() and (y_prob <= 1).all()

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
