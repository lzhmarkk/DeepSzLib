import os
import sys

sys.path.append(os.getcwd())
import h5py
import numpy as np
from tqdm import tqdm
from utils.utils import EarlyStop
from preprocess.utils import compute_FFT
from utils.dataloader import get_dataloader


def load_data(args):
    dataloaders = get_dataloader(args)

    T = args.window // args.seg
    C = args.n_channels
    S = args.seg

    orig_x, fft_x, l = [], [], []
    for loader in dataloaders:
        dataset = loader.dataset
        labels = dataset.labels
        pos_idx = np.where(labels)[0]
        assert len(pos_idx) == dataset.n_pos
        for i in tqdm(pos_idx):
            file_id = i // dataset.n_samples_per_file
            smp_id = i % dataset.n_samples_per_file

            with h5py.File(os.path.join(dataset.path, f"{file_id}.h5"), "r") as hf:
                x = hf['x'][smp_id].transpose(0, 2, 1)
                _l = hf['label'][smp_id].reshape(T, S).any(axis=1)

            orig_x.append(x)
            fft_x.append(fft_and_norm(args, x, T, C, S))
            l.append(_l)

    orig_x = np.stack(orig_x, axis=0)  # (N, T, C, S)
    fft_x = np.stack(fft_x, axis=0)  # (N, T, C, D)
    l = np.stack(l, axis=0)  # (N, T)
    return orig_x, fft_x, l


def load_model(args):
    save_folder = os.path.join('./saves', args.dataset + '-' + args.setting, args.model, args.name)
    saves = list(filter(lambda f: '.pt' in f, os.listdir(save_folder)))[0]
    early_stop = EarlyStop(args, model_path=os.path.join(save_folder, saves))
    model = early_stop.load_best_model(device=args.device)
    model.task = ['onset_detection']
    model.eval()
    print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
    return model


def fft_and_norm(args, x, T, C, S):
    x = x.reshape(T * C, S)
    x = compute_FFT(x, n=S)
    x = x.reshape(T, C, S // 2)
    x = args.scaler.transform(x)
    return x
