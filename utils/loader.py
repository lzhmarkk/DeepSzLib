import importlib
import torch


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
