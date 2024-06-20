import os
import mne
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from process import process_TUSZ, process
from collections import defaultdict
from utils import get_sample_label

origin_dir = f"./data/original_dataset/TUSZ"
dest_dir = f"./data/TUSZ"
channels = ["EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4", "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7",
            "EEG F8", "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ", "EEG PZ", ]
n_sample_per_file = 1000
np.random.seed(0)


def load_edf_data(edf_path, sample_rate):
    data = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
    orig_smp_rate = int(data.info['sfreq'])

    data_channels = []
    for c in channels:
        matched_channels = sorted(list(filter(lambda e: c in e, data.ch_names)))
        matched_channel = matched_channels[0]

        data_channels.append(data[matched_channel][0])

    data = np.concatenate(data_channels, axis=0).T  # (T, C)
    assert data.shape[1] == len(channels)

    resample_data = resample(data, num=data.shape[0] // orig_smp_rate * sample_rate, axis=0)
    return data, resample_data


def load_truth_data(csv_path, length, sample_rate):
    seizure_types = ["BCKG", "FNSZ", "GNSZ", "SPSZ", "CPSZ", "ABSZ", "TNSZ", "CNSZ", "TCSZ", "ATSZ", "MYSZ", "NESZ"]

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
    parser.add_argument("--setting", type=str, choices=["Inductive", "Transductive", 'Classification'], required=True)
    parser.add_argument("--split", type=str, default="7/1/2")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--patch_len", type=int, default=1)
    args = parser.parse_args()

    sample_rate = args.sample_rate
    setting = args.setting
    split = args.split
    window = args.window
    horizon = args.horizon
    stride = args.stride
    patch_len = args.patch_len
    ratio = [float(r) for r in str(split).split('/')]
    ratio = [r / sum(ratio) for r in ratio]

    dest_dir = dest_dir + f'-{setting}'
    shutil.rmtree(dest_dir, ignore_errors=True)

    if setting == 'Transductive':
        user2id = []
        files = []
        for stage in ['train', 'dev', 'eval']:
            for u in os.listdir(os.path.join(origin_dir, 'edf', stage)):
                user2id.append(u)
                for session in os.listdir(os.path.join(origin_dir, 'edf', stage, u)):
                    for subdir in os.listdir(os.path.join(origin_dir, 'edf', stage, u, session)):
                        cur_dir = os.path.join(origin_dir, 'edf', stage, u, session, subdir)
                        for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                            files.append((u, cur_dir, f))

        user2id = {u: i for i, u in enumerate(list(set(user2id)))}

        # load data
        all_u, all_x, all_y = [], [], []
        skip_files = []
        for u, cur_dir, f in tqdm(files):
            try:
                _, x = load_edf_data(os.path.join(cur_dir, f + ".edf"), sample_rate)
                y = load_truth_data(os.path.join(cur_dir, f + ".csv"), length=x.shape[0], sample_rate=sample_rate)
                y[y != 0] = 1
                all_u.append(user2id[u])
                all_x.append(x)
                all_y.append(y)
            except Exception:
                skip_files.append(f)
                print(f"Skip file {f}")

        print(f"Total skip files {len(skip_files)}")
        attribute = process(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, "Transductive",
                            ratio, dest_dir, split, channels, n_sample_per_file)

    else:
        user2id = []
        # load paths
        all_files = defaultdict(list)
        for stage in ['train', 'dev', 'eval']:
            for u in os.listdir(os.path.join(origin_dir, 'edf', stage)):
                user2id.append(u)
                for session in os.listdir(os.path.join(origin_dir, 'edf', stage, u)):
                    for subdir in os.listdir(os.path.join(origin_dir, 'edf', stage, u, session)):
                        cur_dir = os.path.join(origin_dir, 'edf', stage, u, session, subdir)
                        for f in [f.split('.')[0] for f in os.listdir(cur_dir) if '.edf' in f]:
                            all_files[stage].append((u, cur_dir, f))

        user2id = {u: i for i, u in enumerate(list(set(user2id)))}

        # load data
        attribute = {}
        all_classes = set()
        all_u, all_x, all_y = defaultdict(list), defaultdict(list), defaultdict(list)
        for stage in ['train', 'val', 'test']:
            print('\n' + "*" * 30 + stage + "*" * 30 + '\n')
            if stage == 'train':
                files = all_files['train']
            elif stage == 'val':
                files = all_files['dev']
            else:
                files = all_files['eval']

            skip_files = []
            for u, cur_dir, f in tqdm(files, desc=stage):
                try:
                    _, x = load_edf_data(os.path.join(cur_dir, f + ".edf"), sample_rate)
                    y = load_truth_data(os.path.join(cur_dir, f + ".csv"), length=x.shape[0], sample_rate=sample_rate)
                    all_classes.update([get_sample_label(y[i:i + sample_rate * window]) for i in range(0, len(y), sample_rate * window)])
                    all_u[stage].append(user2id[u])
                    all_x[stage].append(x)
                    all_y[stage].append(y)
                except Exception:
                    skip_files.append(f)
                    print(f"Skip file {f}")
            print(f"Total skip files {len(skip_files)} in {stage}")

        # remapping class ids
        classes_mapping = {}
        for i, cls in enumerate(all_classes):
            if setting == 'Classification':
                classes_mapping[cls] = i
            else:
                classes_mapping[cls] = int(cls != 0)  # all seizure is class 1 in detection dataset

        attribute = {}
        shutil.rmtree(dest_dir, ignore_errors=True)
        for stage in ['train', 'val', 'test']:
            for y in all_y[stage]:
                for i in range(len(y)):
                    y[i] = classes_mapping[y[i]]

            attr_updates = process_TUSZ(all_u[stage], all_x[stage], all_y[stage], sample_rate, window, horizon,
                                        stride, patch_len, stage, dest_dir, n_sample_per_file,
                                        classification=setting == 'Classification')
            for k in attr_updates:
                if k == 'n_user' and k in attribute:
                    attribute[k] += attr_updates[k]
                else:
                    attribute[k] = attr_updates[k]

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
            json.dump(attribute, fp, indent=4)
