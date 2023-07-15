import json
import torch
import models
import argparse
import torch.nn as nn


def parse():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--name", type=str, help="Experiment name", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name", default="Seizure")
    parser.add_argument("--device", type=int, help="Device, -1 for cpu", default=-1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1234)
    parser.add_argument("--runs", type=int, help="Number of runs", default=3)
    parser.add_argument("--debug", help="Debug mode", action='store_true')

    parser.add_argument("--patience", type=int, help="Early stop patience", default=40)
    parser.add_argument("--epochs", type=int, help="Maximum epoch", default=200)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=512)
    parser.add_argument("--shuffle", type=bool, help="Shuffle training set", default=True)
    parser.add_argument("--argument", help="Data argument (flip and scale)", action='store_true')
    parser.add_argument("--balance", type=int, help="Balance the training set (n_neg/n_pos)", default=-1)

    # setting
    parser.add_argument("--mode", type=str, help="Training mode: Transductive or Inductive", default='Transductive')
    parser.add_argument("--preprocess", type=str, help="seg or fft", default='fft')
    parser.add_argument("--split", type=str, help="Percentile to split train/val/test sets", default="7/1/2")
    parser.add_argument("--norm", type=bool, help="Z-normalizing data", default=True)

    parser.add_argument("--window", type=int, help="Look back window (second)", default=30)
    parser.add_argument("--horizon", type=int, help="Future predict horizon", default=30)
    parser.add_argument("--stride", type=int, help="Window moving stride (second)", default=30)
    parser.add_argument("--seg", type=int, help="Segment length (seconds)", default=1)

    # training
    parser.add_argument("--pred_loss", type=str, help="Prediction loss function", default="MSE")
    parser.add_argument("--cls_loss", type=str, help="Classification loss function", default="BCE")
    parser.add_argument("--multi_task", help="Use multi-task", action='store_true')
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

    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    return args


def get_model(args):
    model = getattr(models, args.model)
    model = model(args)
    return model


def get_optimizer(args, param):
    optimizer = args.optim

    if optimizer == 'SGD':
        return torch.optim.SGD(params=param, lr=args.lr, weight_decay=args.wd)
    elif optimizer == 'Adam':
        return torch.optim.Adam(params=param, lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Not implemented optimizer: {optimizer}")


def get_scheduler(args, optim):
    assert isinstance(optim, torch.optim.Optimizer)
    scheduler = args.scheduler

    if scheduler == 'Exp':
        return torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)
    elif scheduler == 'Step':
        return torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.7)
    elif scheduler == 'Cosine':
        t_max = len(args.dataset['train']) / args.batch_size * args.patience / 3
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max)
    elif scheduler == 'None':
        class EmptyScheduler:
            def __init__(self, lr):
                self.lr = lr

            def get_lr(self):
                return [self.lr]

            def get_last_lr(self):
                return [self.lr]

            def step(self):
                pass

        return EmptyScheduler(args.lr)
    else:
        raise ValueError(f"Not implemented scheduler: {scheduler}")


def parse_model_config(args, model):
    with open(f"./models/{model}/config.json", 'r') as f:
        model_cfg = json.load(f)

    for k, v in model_cfg.items():
        setattr(args, k, v)
    return args
