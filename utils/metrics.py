import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, fbeta_score
from collections import defaultdict


def correct_score(truth, pred, thres=1):
    """
    The rate of correctly predict events within several seconds after truth events happen
    """
    metric = {}
    pred_horizons = defaultdict(list)
    for _truth, _pred in zip(truth, pred):
        if _truth.any():
            onset = _truth.nonzero()[0].min()
            for delta in range(1, truth.shape[1] - onset + 1):
                _truth_delta = _truth[onset:onset + delta].any()
                _pred_delta = np.sum(_pred[onset:onset + delta]) >= thres
                if _truth_delta:
                    pred_horizons[delta].append(_pred_delta)

    for delta in range(1, truth.shape[1] + 1):
        pred = np.array(pred_horizons[delta])
        correct_rate = np.sum(pred == 1) / len(pred)
        metric[f'correct-{delta}'] = correct_rate
        # metric[f'miss-{delta}'] = 1 - correct_rate

    return metric


def wrong_score(truth, pred, thres=1):
    """
    The rate of mistakenly predict events when no event will happen in the next several seconds
    """
    metric = {}
    truth_horizons = defaultdict(list)
    for _truth, _pred in zip(truth, pred):
        pred_events = _pred.nonzero()[0]
        for onset in pred_events:
            for delta in range(1, truth.shape[1] - onset + 1):
                _truth_delta = _truth[onset:onset + delta].any()
                _pred_delta = np.sum(_pred[onset:onset + delta]) >= thres
                if _pred_delta:
                    truth_horizons[delta].append(_truth_delta)

    for delta in range(1, truth.shape[1] + 1):
        truth = np.array(truth_horizons[delta])
        wrong_score_delta = 1 - np.sum(truth) / len(truth)
        metric[f'wrong-{delta}'] = wrong_score_delta

    return metric


def get_detection_metrics(prob, truth, threshold_value=0.5):
    assert (0 <= prob).all() and (prob <= 1).all()
    assert prob.shape == truth.shape
    assert prob.ndim == 1 and truth.ndim == 1

    pred = (prob >= threshold_value).astype(int)

    return {
        'accuracy': accuracy_score(truth, pred),
        'precision': precision_score(truth, pred, average='binary'),
        'recall': recall_score(truth, pred, average='binary'),
        'f1': f1_score(truth, pred, average='binary'),
        'f2': fbeta_score(truth, pred, beta=2, average='binary'),
        'auc': roc_auc_score(truth, prob)
    }


def get_onset_detection_metrics(prob, truth, threshold_value=0.5):
    assert (0 <= prob).all() and (prob <= 1).all()
    assert prob.shape == truth.shape
    assert prob.ndim == 2 and truth.ndim == 2

    pred = (prob >= threshold_value).astype(int)

    truth_flat = truth.flatten()
    pred_flat = pred.flatten()

    correct_rate = correct_score(truth, pred, thres=1)
    wrong_rate = wrong_score(truth, pred, thres=1)

    metric = {
        'accuracy': accuracy_score(truth_flat, pred_flat),
        'precision': precision_score(truth_flat, pred_flat, average='binary'),
        'recall': recall_score(truth_flat, pred_flat, average='binary'),
        'f1': f1_score(truth_flat, pred_flat, average='binary'),
        'f2': fbeta_score(truth_flat, pred_flat, beta=2, average='binary'),
        'auc': roc_auc_score(truth_flat, pred_flat)
    }
    metric.update(correct_rate)
    metric.update(wrong_rate)

    return metric


def get_classification_metrics(prob, truth):
    assert (0 <= prob).all() and (prob <= 1).all()
    assert prob.ndim == 2 and truth.ndim == 1

    metric = {'f1_weighted': f1_score(truth, np.argmax(prob, axis=1), average='weighted')}
    for i, m in enumerate(f1_score(truth, np.argmax(prob, axis=1), average=None)):
        metric[f'f1_{i}'] = m

    metric['auc_weighted'] = roc_auc_score(truth, prob, average='weighted', multi_class='ovr', labels=range(prob.shape[1]))
    for i, m in enumerate(roc_auc_score(truth, prob, average=None, multi_class='ovr', labels=range(prob.shape[1]))):
        metric[f'auc_{i}'] = m

    return metric


def thresh_max_f1(y_true, y_prob):
    """
    Find the best threshold based on precision-recall curve to maximize F1-score.
    Binary classification only
    """
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()
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
