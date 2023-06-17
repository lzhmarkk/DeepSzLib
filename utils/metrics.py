from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_metrics(pred, truth):
    metric = {}
    # todo more metrics
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
