import json
import torch
import argparse
from models import *
import torch.nn as nn


def parse():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--exp_id", type=str, help="Experiment name", required=True)
    parser.add_argument("--model", type=str, help="Model name", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name", default="Seizure")
    parser.add_argument("--device", type=int, help="Device, -1 for cpu", default=-1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1234)
    parser.add_argument("--patience", type=int, help="Early stop patience", default=40)
    parser.add_argument("--epochs", type=int, help="Maximum epoch", default=200)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("--shuffle", type=bool, help="Shuffle training set", default=True)

    # setting
    parser.add_argument("--mode", type=str, help="Training mode: Transductive or Inductive", default='Transductive')
    parser.add_argument("--task", type=str, help="Training task: Prediction (pred), classification (cls) or both", default='cls')
    parser.add_argument("--window", type=int, help="Look back window (second)", default=30)
    parser.add_argument("--horizon", type=int, help="Future predict horizon", default=30)
    parser.add_argument("--stride", type=int, help="Window moving stride (second)", default=15)
    parser.add_argument("--seg", type=int, help="Segment length (seconds), -1 for no-segmentation", default=1)
    parser.add_argument("--split", type=str, help="Percentile to split train/val/test sets", default="7/2/1")
    parser.add_argument("--sigma", type=int, help="Data out of [μ-3σ, μ-3σ] will be dropped", default=3)

    # training
    parser.add_argument("--pred_loss", type=str, help="Prediction loss function", default="MSE")
    parser.add_argument("--cls_loss", type=str, help="Classification loss function", default="BCE")
    parser.add_argument("--optim", type=str, help="Optimizer", default='Adam')
    parser.add_argument("--scheduler", type=str, help="Scheduler", default='None')
    parser.add_argument("--reduction", type=str, help="Reduction of loss function", default='mean')
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--wd", type=str, help="Weight decay", default=5e-4)

    args = parser.parse_args()
    args = parse_model_config(args, args.model)
    args.backward = True  # default

    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    return args


def get_model(args):
    task = args.task
    model = args.model

    if model == 'LOF':
        model = LOF(args)
    else:
        raise ValueError(f"Not implemented model: {model}")
    return model


def get_loss(args):
    task = args.task
    assert task == 'cls', f"Prediction task not implemented"
    loss = args.cls_loss

    if loss == 'MAE':
        return nn.L1Loss(reduction=args.reduction)
    elif loss == 'MSE':
        return nn.MSELoss(reduction=args.reduction)
    elif loss == 'BCE':
        return nn.BCELoss(reduction=args.reduction)
    else:
        raise ValueError(f"Not implemented loss: {loss}")


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
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    elif scheduler == 'None':
        class EmptyScheduler:
            def __init__(self, lr):
                self.lr = lr

            def get_lr(self):
                return self.lr

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
