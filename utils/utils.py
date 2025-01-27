import os
import sys
import torch
import shutil
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Timer:
    def __init__(self):
        self.flush()

    def tick(self, name):
        if self.last is not None:
            delta = datetime.now() - self.last
            delta = delta.seconds + delta.microseconds / 1e6
            self.itv[name].append(delta)
        self.last = datetime.now()

    def flush(self):
        self.itv = defaultdict(list)
        self.last = datetime.now()

    def get(self, name):
        return self.itv[name][-1]

    def get_all(self, name):
        return np.mean(self.itv[name]).item()


class EarlyStop:
    def __init__(self, args, model_path):
        self.patience = args.patience
        self.best_epoch = 1e5
        self.model_path = model_path

        self.metric = args.metric
        print(f"Use {self.metric} for early stop")
        self.small_better_metrics = ['auc', 'f1', 'f1_weighted', 'auc_weighted']
        if self.metric in self.small_better_metrics:
            self.history_metric = [0]
        else:
            self.history_metric = [1e5]

    def step(self, epoch, loss, scores, model):
        if self.metric in self.small_better_metrics:
            metric = scores[self.metric]
            better = metric > np.max(self.history_metric)
        else:
            metric = loss
            better = metric < np.min(self.history_metric)

        if better:
            with open(self.model_path, 'wb') as fp:
                torch.save(model.state_dict(), fp)
            self.best_epoch = epoch
            print(f'Save best epoch: {self.best_epoch}')

        self.history_metric.append(metric)

    def now_stop(self, epoch):
        if epoch - self.best_epoch > self.patience:
            print(f'Early Stop after {self.patience} epochs. Best epoch: {self.best_epoch}')
            return True
        else:
            return False

    def load_best_model(self, model=None, device=None):
        if device is not None:
            device = torch.device(device)

        with open(self.model_path, 'rb') as fp:
            if model is not None:
                model.load_state_dict(torch.load(fp, device))
            else:
                model = torch.load(fp, device)

        if self.metric in self.small_better_metrics:
            best_epoch = np.argmax(self.history_metric)
        else:
            best_epoch = np.argmin(self.history_metric)
        print("Training finished")
        print('Best epoch:', best_epoch - 1)  # -1 to skip the first value 1e5
        print("The validation {} on best model is {:.4f}".format(self.metric, self.history_metric[best_epoch]))
        return model


class Scaler:
    def __init__(self, mean, std, norm):
        self.mean = []
        self.std = []
        self.norm = norm

        if norm:
            if isinstance(mean, list):
                self.mean = np.expand_dims(np.array(mean), axis=-1)
                self.std = np.expand_dims(np.array(std), axis=-1)
            else:
                self.mean = mean
                self.std = std
        else:
            self.mean = 0
            self.std = 1

    def transform(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x_transformed = [self.transform(_x) for _x in x]
        else:
            x_transformed = (x - self.mean) / self.std
        return x_transformed

    def inv_transform(self, x):
        if isinstance(x, torch.Tensor):
            x_inv_transformed = x * torch.from_numpy(self.std).to(x.device) + torch.from_numpy(self.mean).to(x.device)
        else:
            x_inv_transformed = []
            for _x in x:
                _x = _x * self.std + self.mean
                x_inv_transformed.append(_x)

        return x_inv_transformed


def to_gpu(*data, device):
    res = []
    for item in data:
        if isinstance(item, tuple) or isinstance(item, list):
            item = to_gpu(*item, device=device)
        elif isinstance(item, np.ndarray):
            item = torch.from_numpy(item).float().to(device)
        elif isinstance(item, torch.Tensor):
            item = item.float().to(device)
        res.append(item)
    return tuple(res)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(6)


def init_env(save_folder):
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))


def init_run_env(args, run_id):
    run_folder = os.path.join(args.save_folder, f'run-{run_id}')
    os.makedirs(run_folder, exist_ok=True)
    args.writer = SummaryWriter(run_folder)
    args.timer = Timer()
    args.early_stop = EarlyStop(args, model_path=os.path.join(args.save_folder, f'best-model-{run_id}.pt'))
