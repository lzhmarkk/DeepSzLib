import os
import sys

sys.path.append(os.getcwd())
import torch
import numpy as np
from tqdm import tqdm
from utils.parser import parse
from utils.utils import set_random_seed, to_gpu
from visualize.util import fft_and_norm, load_data, load_model


def gen_occlusion_map():
    model.task = 'detection'
    occlusion_map = np.zeros((N, T, C))
    for i, x in enumerate(tqdm(orig_x, ncols=150)):
        occlusion_x = np.expand_dims(x, axis=[0, 1]).repeat(T, axis=0).repeat(C, axis=1)  # (T, C, T, C, S)
        for t in range(T):
            for c in range(C):
                occlusion_x[t, c, t, c] = 0
        x = np.concatenate([np.expand_dims(x, axis=0), occlusion_x.reshape(T * C, T, C, D)], axis=0)
        x = np.stack([fft_and_norm(_x, T, C, S) for _x in x], axis=0)

        with torch.no_grad():
            x = to_gpu(x, device=args.device)[0]
            z = model(x, None, None)  # (1+T*C)

        z = torch.sigmoid(z[0]).cpu().numpy()
        orig_z = z[0]
        occlusion_z = z[1:].reshape(T, C) - orig_z
        occlusion_z = np.abs(occlusion_z)
        occlusion_z = (occlusion_z - occlusion_z.min()) / (occlusion_z.max() - occlusion_z.min())
        occlusion_map[i] = occlusion_z
    np.savez("./occlusion.npz", occlusion_map=occlusion_map)
    print("occlusion done")


def plot_occlusion_map():
    pass


if __name__ == '__main__':
    args = parse()
    args.task = ['onset_detection']
    set_random_seed(args.seed)
    print(args)

    orig_x, fft_x, l = load_data(args)
    model = load_model(args)

    N = args.n_pos_train + args.n_pos_val + args.n_pos_test
    T = args.window // args.patch_len
    C = args.n_channels
    D = args.input_dim
    H = args.hidden
    S = args.patch_len
    B = args.batch_size

    gen_occlusion_map()
    plot_occlusion_map()
