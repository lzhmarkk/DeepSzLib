import os
import mne
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from process import process_TUSZ
from preprocess.detection_TUSZ import channels, origin_dir, load_edf_data

dest_dir = f"./data/TUSZ"
n_sample_per_file = 1000
np.random.seed(0)


def load_truth_data(csv_path, length, sample_rate):
    seizure_types = ["BCKG", "FNSZ", "GNSZ""SPSZ", "CPSZ", "ABSZ", "TNSZ", "CNSZ", "TCSZ", "ATSZ", "MYSZ", "NESZ"]

    truth = np.zeros([length], dtype=int)

    df = pd.read_csv(csv_path, header=0, comment='#')
    for i, line in df.iterrows():
        if line['label'] != 'bckg':
            s_time = line['start_time']
            e_time = line['stop_time']
            s_time = int(s_time * sample_rate)
            e_time = int(e_time * sample_rate)
            truth[s_time:e_time] = seizure_types.index(line['label'].upper())

    return truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--patch_len", type=int, default=1)
    args = parser.parse_args()

    sample_rate = args.sample_rate
    window = args.window
    horizon = args.horizon
    stride = args.stride
    patch_len = args.patch_len

    dest_dir = dest_dir + f'-Classification'
    user2id = []
    # load paths
    train_files, val_files, test_files = [], [], []
    for u in os.listdir(os.path.join(origin_dir, 'edf', 'train')):
        user2id.append(u)
        for session in os.listdir(os.path.join(origin_dir, 'edf', 'train', u)):
            for subdir in os.listdir(os.path.join(origin_dir, 'edf', 'train', u, session)):
                cur_dir = os.path.join(origin_dir, 'edf', 'train', u, session, subdir)
                for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                    train_files.append((u, cur_dir, f))
    for u in os.listdir(os.path.join(origin_dir, 'edf', 'dev')):
        user2id.append(u)
        for session in os.listdir(os.path.join(origin_dir, 'edf', 'dev', u)):
            for subdir in os.listdir(os.path.join(origin_dir, 'edf', 'dev', u, session)):
                cur_dir = os.path.join(origin_dir, 'edf', 'dev', u, session, subdir)
                for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                    val_files.append((u, cur_dir, f))
    for u in os.listdir(os.path.join(origin_dir, 'edf', 'eval')):
        user2id.append(u)
        for session in os.listdir(os.path.join(origin_dir, 'edf', 'eval', u)):
            for subdir in os.listdir(os.path.join(origin_dir, 'edf', 'eval', u, session)):
                cur_dir = os.path.join(origin_dir, 'edf', 'eval', u, session, subdir)
                for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                    test_files.append((u, cur_dir, f))

    user2id = {u: i for i, u in enumerate(list(set(user2id)))}

    # load data
    attribute = {}
    for stage in ['train', 'val', 'test']:
        print('\n' + "*" * 30 + stage + "*" * 30 + '\n')
        all_u, all_x, all_y = [], [], []
        if stage == 'train':
            files = train_files
        elif stage == 'val':
            files = val_files
        else:
            files = test_files

        skip_files = []
        for u, cur_dir, f in tqdm(files, desc=stage):
            try:
                _, x = load_edf_data(os.path.join(cur_dir, f + ".edf"), sample_rate)
                y = load_truth_data(os.path.join(cur_dir, f + ".csv"), length=x.shape[0], sample_rate=sample_rate)
                all_u.append(user2id[u])
                all_x.append(x)
                all_y.append(y)
            except Exception:
                skip_files.append(f)
                print(f"Skip file {f}")

        print(f"Total skip files {len(skip_files)} in {stage}")
        attribute = process_TUSZ(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, stage, dest_dir, n_sample_per_file,
                                 attribute, drop_neg_samples=True)

    # config
    with open(os.path.join(dest_dir, "./config.json"), 'w') as fp:
        config = {'window': window, 'horizon': horizon, 'stride': stride, 'patch_len': patch_len}
        json.dump(config, fp, indent=2)

    # attribute
    with open(os.path.join(dest_dir, "./attribute.json"), 'w') as fp:
        attribute['sample_rate'] = sample_rate
        attribute['n_samples_per_file'] = n_sample_per_file
        attribute["n_channels"] = len(channels)
        attribute["channels"] = channels
        json.dump(attribute, fp, indent=2)
