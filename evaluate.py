import os
import json
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics, thresh_max_f1
from utils.parser import parse
from utils.loss import MyLoss
from utils.dataloader import get_dataloader
from utils.utils import Timer, EarlyStop, set_random_seed, to_gpu


def evaluate(args, stage, model, loss, loader):
    pred, real, eval_loss = [], [], []
    tqdm_loader = tqdm(loader, ncols=150)
    for i, (u, x, y, p) in enumerate(tqdm_loader):
        model.eval()
        with torch.no_grad():
            x, y, p = to_gpu(x, y, p, device=args.device)
            z = model(x, p, y)
            los = loss(z, p, y)

        pred.append(z[0])
        real.append(p)
        eval_loss.append(los.item())

    eval_loss = np.mean(eval_loss).item()
    pred = torch.sigmoid(torch.cat(pred, dim=0)).cpu().numpy()
    real = torch.cat(real, dim=0).cpu().numpy()

    if args.threshold:
        if stage == 'train':
            args.threshold_value = 0.5
        elif stage == 'val':
            threshold_value = thresh_max_f1(y_true=real, y_prob=pred)
            args.threshold_value = threshold_value
    else:
        args.threshold_value = 0.5

    scores = get_metrics(pred, real, threshold_value=args.threshold_value)
    return eval_loss, scores, pred, real


if __name__ == '__main__':
    args = parse()
    set_random_seed(args.seed)
    print(args)

    # save folder
    save_folder = os.path.join('./saves', args.dataset, args.model, args.expid)
    run_folder = os.path.join(save_folder, 'run')
    timer = Timer()
    early_stop = EarlyStop(args, model_path=os.path.join(save_folder, 'best-model.pt'))

    # read model
    model = early_stop.load_best_model()
    print(model)
    print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
    loss = MyLoss(args)

    # load data
    train_loader, val_loader, test_loader = get_dataloader(args)

    # validate model
    _, valid_scores, _, _ = evaluate(args, 'val', model, loss, val_loader)

    # test model
    _, test_scores, pred, tgt = evaluate(args, 'test', model, loss, test_loader)
    print('Test results:')
    print(json.dumps(test_scores, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=pred, targets=tgt)
