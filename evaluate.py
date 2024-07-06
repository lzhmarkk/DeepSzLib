import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import (get_detection_metrics,
                           get_onset_detection_metrics,
                           get_classification_metrics,
                           get_prediction_metrics,
                           thresh_max_f1)
from utils.parser import parse
from utils.loss import MyLoss
from utils.dataloader import get_dataloader
from utils.utils import EarlyStop, set_random_seed, to_gpu


def evaluate(args, stage, model, loss, loader):
    prob, label = [], []  # detection
    pred, future = [], []  # prediction
    eval_loss = []

    tqdm_loader = tqdm(loader, ncols=150)
    for i, (u, x, y, p) in enumerate(tqdm_loader):
        model.eval()
        with torch.no_grad():
            x, y, p = to_gpu(x, y, p, device=args.device)
            z = model(x, p, y)
            los = loss(z, p, y)

        prob.append(z['prob'].cpu())
        label.append(p.cpu())
        pred.append(z['pred'].cpu() if 'pred' in z else None)
        future.append(y.cpu() if y else None)
        eval_loss.append(los.item())

    if 'detection' in args.task or 'onset_detection' in args.task:
        prob = torch.sigmoid(torch.cat(prob, dim=0)).numpy()
        label = torch.cat(label, dim=0).numpy()

        if stage == 'train':
            args.threshold_value = 0.5
        elif stage == 'val':
            if args.threshold:
                args.threshold_value = float(args.threshold)
            else:
                args.threshold_value = thresh_max_f1(y_true=label, y_prob=prob)
            print(f"Use threshold {args.threshold_value}")

        if 'detection' in args.task:
            scores = get_detection_metrics(prob, label, threshold_value=args.threshold_value)
        else:
            scores = get_onset_detection_metrics(prob, label, threshold_value=args.threshold_value)

    elif 'classification' in args.task:
        args.threshold_value = None
        prob = torch.softmax(torch.cat(prob, dim=0), dim=-1).numpy()
        label = torch.cat(label, dim=0).numpy()

        scores = get_classification_metrics(prob, label)

    elif 'prediction' in args.task:
        pred = torch.cat(pred, dim=0).numpy()
        future = torch.cat(future, dim=0).numpy()

        scores = get_prediction_metrics(pred, future)
        prob, label = pred, future

    else:
        raise ValueError()

    return np.mean(eval_loss).item(), scores, prob, label


if __name__ == '__main__':
    args = parse()
    set_random_seed(args.seed)
    print(args)

    _, val_loader, test_loader = get_dataloader(args)

    test_scores_multiple_runs = []
    saves = list(filter(lambda f: '.pt' in f, os.listdir(args.save_folder)))
    for run in range(len(saves)):
        early_stop = EarlyStop(args, model_path=os.path.join(args.save_folder, f'best-model-{run}.pt'))

        # read model
        model = early_stop.load_best_model()
        print(model)
        print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
        loss = MyLoss(args)

        _, _, _, _ = evaluate(args, 'val', model, loss, test_loader)
        # test model
        _, test_scores, pred, tgt = evaluate(args, 'test', model, loss, test_loader)

        test_scores_multiple_runs.append(test_scores)

    # merge results from several runs
    test_scores = {'mean': {}, 'std': {}}
    for k in test_scores_multiple_runs[0].keys():
        test_scores['mean'][k] = np.mean([scores[k] for scores in test_scores_multiple_runs]).item()
        test_scores['std'][k] = np.std([scores[k] for scores in test_scores_multiple_runs]).item()

    print(f"Dataset: {args.dataset}, model: {args.model}, setting: {args.setting}")
    print('*' * 30, 'mean', '*' * 30)
    skip_keys = lambda k: '-' in str(k) and int(str(k).split('-')[-1]) not in [5, 10, 15]
    for k in test_scores['mean']:
        if not skip_keys(k):
            print(f"{k}\t", end='')
    print()
    for k in test_scores['mean']:
        if not skip_keys(k):
            print("{:.4f}\t".format(test_scores['mean'][k]), end='')
    print()

    print('*' * 30, 'std', '*' * 30)
    for k in test_scores['std']:
        if not skip_keys(k):
            print(f"{k}\t", end='')
    print()
    for k in test_scores['std']:
        if not skip_keys(k):
            print("{:.4f}\t".format(test_scores['std'][k]), end='')
    print()
