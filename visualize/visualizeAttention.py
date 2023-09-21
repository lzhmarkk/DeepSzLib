import os
import sys

sys.path.append(os.getcwd())
import math
import torch
import numpy as np
from tqdm import tqdm
from utils.parser import parse
import matplotlib.pyplot as plt
from utils.utils import set_random_seed, to_gpu
from visualize.util import load_data, load_model


def gen_attention_map():
    if os.path.exists("./attn.npz"):
        return

    setattr(model.local_gnn[0], 'attn_weight', [])
    for b in tqdm(range(math.ceil(N / B))):
        x = fft_x[b * B:(b + 1) * B]
        with torch.no_grad():
            x = to_gpu(x, device=args.device)[0]
            _ = model(x, None, None)

    attn_weight = model.local_gnn[0].attn_weight
    attn_weight = torch.cat([batch_attn_weight.reshape(-1, C, T).transpose(1, 2) for batch_attn_weight in attn_weight], dim=0)
    attn_weight = attn_weight.reshape(N, T, C)
    np.savez("./attn.npz", attn_weight=attn_weight.cpu().numpy())
    delattr(model.local_gnn[0], 'attn_weight')
    print("attn done")


def plot_attention_map(index=None, skip_head=2, topk=10):
    os.makedirs("./visualize_attention", exist_ok=True)
    data = np.load("./attn.npz", allow_pickle=True)

    attn_weight = data['attn_weight']  # (N, T, C)

    if index is None:
        attn_score = np.expand_dims(l, axis=2) * attn_weight
        attn_score_sum = attn_score.sum(axis=(1, 2))
        sort_idx = np.argsort(attn_score_sum)[::-1]
    else:
        sort_idx = index

    for i in tqdm(sort_idx[:topk]):
        fig, axs = plt.subplots(1, sharex='all', figsize=(T, C))
        attn_w = attn_weight[i].T
        attn_w = (attn_w - attn_w.min()) / (attn_w.max() - attn_w.min())
        attn_w = attn_w[:, skip_head:]
        for c in range(0, C):
            y = orig_x[i, :, c, :].reshape(T * S)[skip_head * S:]
            y = (y - y.min()) / (y.max() - y.min())  # norm to 0-1
            y = y + C - 1 - c
            axs.plot(range((T - skip_head) * S), y, color='black')
        axs.imshow(attn_w,
                   aspect='auto',
                   extent=[0, (T - skip_head) * S, 0, C],
                   origin='lower',
                   interpolation='bilinear',
                   cmap='hot',
                   alpha=0.5)
        axs.set_xticks(range(0, (T - skip_head) * S, S), l[i].astype(int)[skip_head:])

        plt.tight_layout()
        plt.savefig(os.path.join("./visualize_attention", f"{i}.png"), dpi=300)
        plt.show()


if __name__ == '__main__':
    args = parse()
    args.task = ['anomaly']
    set_random_seed(args.seed)
    print(args)

    orig_x, fft_x, l = load_data(args)
    model = load_model(args)

    mean = args.mean['seg']
    N = args.n_pos_train + args.n_pos_val + args.n_pos_test
    T = args.window // args.seg
    C = args.n_channels
    D = args.input_dim
    H = args.hidden
    S = args.seg
    B = args.batch_size

    gen_attention_map()
    plot_attention_map(index=[336, 1407, 2792, 3366, 5503, 7414, 8379, 8380, 10311], topk=100)
