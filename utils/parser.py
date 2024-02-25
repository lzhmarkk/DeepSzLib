import json
import torch
import models
import argparse
import importlib


def parse():
    parser = argparse.ArgumentParser()

    # pretrain
    parser.add_argument("--pretrain", action='store_true')

    # model
    parser.add_argument("--hidden", type=int, help="Hidden dimension", required=True)

    # experiment
    parser.add_argument("--name", type=str, help="Experiment name", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name", default="FDUSZ")
    parser.add_argument("--device", type=int, help="Device, -1 for cpu", default=-1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1234)
    parser.add_argument("--runs", type=int, help="Number of runs", default=5)
    parser.add_argument("--n_worker", type=int, help="Number of dataloader workers", default=8)
    parser.add_argument("--debug", help="Debug mode", action='store_true')
    parser.add_argument("--threshold", type=float, help="Decision threshold. None for auto", default=None)
    parser.add_argument("--metric", help="Early stop metric", choices=['auc', 'f1', 'loss'], default='auc')
    parser.add_argument("--pin_memory", help="Load all data into memory", action='store_true')

    parser.add_argument("--patience", type=int, help="Early stop patience", default=20)
    parser.add_argument("--epochs", type=int, help="Maximum epoch", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=256)
    parser.add_argument("--shuffle", type=bool, help="Shuffle training set", default=True)
    parser.add_argument("--argument", help="Data argument (flip and scale)", action='store_true')
    parser.add_argument("--balance", type=int, help="Balance the training set (n_neg/n_pos)", default=1)

    # setting
    parser.add_argument("--setting", type=str, choices=['Transductive', 'Inductive'], default='Transductive')
    parser.add_argument('--task', type=str, nargs='+', help='Task', choices=['cls', 'anomaly', 'pred'], default=['cls'])
    parser.add_argument('--anomaly_len', type=int, default=15)
    parser.add_argument("--cls_loss", type=str, help="Classification loss function", default="BCE")
    parser.add_argument("--anomaly_loss", type=str, help="Anomaly loss function", default="BCE")
    parser.add_argument("--pred_loss", type=str, help="Prediction loss function", default="MSE")

    parser.add_argument("--preprocess", type=str, help="raw or fft", default='fft')
    parser.add_argument("--split", type=str, help="Percentile to split train/val/test sets", default="7/1/2")
    parser.add_argument("--no_norm", help="Do NOT use z-normalizing", action='store_true')
    parser.add_argument("--window", type=int, help="Look back window (second)", default=30)
    parser.add_argument("--horizon", type=int, help="Future predict horizon", default=30)
    parser.add_argument("--stride", type=int, help="Window moving stride (second)", default=30)
    parser.add_argument("--seg", type=float, help="Segment length (seconds)", default=1)

    # training
    parser.add_argument("--lamb", type=float, help="L_{cls}+λ*L_{pred}", default=1.0)
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--scheduler", type=str, help="Scheduler", default='Cosine')
    parser.add_argument("--grad_clip", type=float, help="Gradient clip", default=5.0)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="Weight decay", default=5e-4)

    args = parser.parse_args()
    args = parse_model_config(args, args.model)
    args.backward = True  # default. Set false for not-training methods
    args.data_loaded = False
    assert not ('cls' in args.task and 'anomaly' in args.task)

    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    return args


def get_model(args):
    context = importlib.import_module(f'models.{args.model}.{args.model}')
    model = getattr(context, args.model)
    model = model(args)
    return model


def get_optimizer(args, param):
    if not args.backward:
        return None

    optimizer = args.optim

    if optimizer == 'SGD':
        return torch.optim.SGD(params=param, lr=args.lr, weight_decay=args.wd)
    elif optimizer == 'Adam':
        return torch.optim.Adam(params=param, lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Not implemented optimizer: {optimizer}")


def get_scheduler(args, optim):
    class EmptyScheduler:
        def __init__(self, lr):
            self.lr = lr

        def get_lr(self):
            return [self.lr]

        def get_last_lr(self):
            return [self.lr]

        def step(self):
            pass

    if not args.backward:
        return EmptyScheduler(args.lr)

    assert isinstance(optim, torch.optim.Optimizer)
    scheduler = args.scheduler

    if scheduler == 'Exp':
        return torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)
    elif scheduler == 'Step':
        return torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.7)
    elif scheduler == 'Cosine':
        t_max = len(args.data['train']) / args.batch_size * args.patience / 3
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max)
    elif scheduler == 'None':
        return EmptyScheduler(args.lr)
    else:
        raise ValueError(f"Not implemented scheduler: {scheduler}")


def parse_model_config(args, model):
    with open(f"./models/{model}/config.json", 'r') as f:
        model_cfg = json.load(f)

    data_name = args.dataset
    setting_name = args.setting
    task = args.task[0]

    match_patterns = [data_name + '-' + setting_name + '-' + task,
                      data_name + '-' + setting_name,
                      data_name,
                      setting_name,
                      'else']

    for k, v in model_cfg.items():
        if isinstance(v, dict):
            for pattern in match_patterns:
                if pattern in v:
                    v = v[pattern]
                    break
        setattr(args, k, v)

    return args
