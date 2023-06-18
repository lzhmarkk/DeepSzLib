import sys
import torch
import random
import numpy as np
from datetime import datetime
from collections import defaultdict


class Logger:
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

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
            self.itv[name].append((datetime.now() - self.last).seconds)
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
        self.history_loss = [1e5]
        self.best_epoch = 1e5
        self.model_path = model_path

    def step(self, epoch, loss, model):
        if loss < np.min(self.history_loss):
            with open(self.model_path, 'wb') as fp:
                torch.save(model, fp)
            self.best_epoch = epoch
            print(f'Save best epoch: {self.best_epoch}')

        self.history_loss.append(loss)

    def now_stop(self, epoch):
        if epoch - self.best_epoch > self.patience:
            print(f'Early Stop after {self.patience} epochs. Best epoch: {self.best_epoch}')
            return True
        else:
            return False

    def load_best_model(self):
        with open(self.model_path, 'rb') as fp:
            model = torch.load(fp)

        best_epoch = np.argmin(self.history_loss)
        print("Training finished")
        print('Best epoch:', best_epoch - 1)  # -1 to skip the first value 1e5
        print("The valid loss on best model is {:.4f}".format(self.history_loss[best_epoch]))
        return model


class Scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        return (x - self.mean) / self.std

    def inv_transform(self, x):
        return x * self.std + self.mean


def to_gpu(*data, device):
    res = []
    for item in data:
        if isinstance(item, tuple):
            item = tuple(map(lambda ele: ele.to(device), item))
        elif isinstance(item, list):
            item = list(map(lambda ele: ele.to(device), item))
        else:
            item = item.to(device)
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
