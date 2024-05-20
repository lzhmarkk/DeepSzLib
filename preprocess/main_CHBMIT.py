import os
import re
import mne
import argparse
import numpy as np
from scipy.signal import resample
from process import process

origin_dir = f"./data/original_dataset/CHBMIT/1.0.0"
dest_dir = f"./data/CHBMIT"
channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8",
            "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
n_sample_per_file = 1000
np.random.seed(0)
drop_non_seizure_files = True


def load_edf_data(edf_path, sample_rate):
    data = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    orig_sample_rate = int(data.info['sfreq'])

    new_data = []
    for c in channels:
        if c in data.ch_names:
            new_data.append(data[c][0])
        elif c + '-0' in data.ch_names:
            new_data.append(data[c + '-0'][0])

    new_data = np.concatenate(new_data, axis=0).T  # (T, C)
    assert new_data.shape[1] == len(channels), f"{edf_path} misses channels"

    resample_data = resample(new_data, num=new_data.shape[0] // orig_sample_rate * sample_rate, axis=0)
    return resample_data


def load_truth_data(extracted_info, length, sample_rate):
    truth = np.zeros([length], dtype=float)

    timestamps = extracted_info['timestamp']
    assert len(timestamps) == 2 * int(extracted_info['Number of Seizures in File'])

    for i in range(0, len(timestamps), 2):
        s_time = int(timestamps[i].strip().split(' ')[0])
        e_time = int(timestamps[i + 1].strip().split(' ')[0])
        s_time *= sample_rate
        e_time *= sample_rate
        assert 0 <= s_time and e_time <= length, f"{extracted_info['File Name']}"
        truth[s_time:e_time] = 1

    return truth


def load_summary(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as fp:
        content = fp.read()
        blocks = content.strip().split('\n\n')

        extracted_info = {}
        current_channels = None
        for block in blocks:
            if block.startswith('Data Sampling Rate'):
                sample_rate = int(re.findall(r'Data Sampling Rate: (\d*) Hz', block)[0])
            elif block.startswith('Channel'):
                channels_in_block = []
                for line in block.split('\n'):
                    match = re.findall(r'Channel (\d{1,2})?: ([A-Za-z\d]+-[A-Za-z\d]+)', line)
                    if match:
                        channels_in_block.append(match[0][1])

                channels_in_block = set(channels_in_block)
                if current_channels is None:
                    current_channels = channels_in_block

            elif block.startswith('File Name'):
                info_dict = {}
                lines = block.strip().split('\n')
                timestamps = []
                for line in lines:
                    key, value = line.split(': ', 1)
                    if re.findall(r"Seizure(.*)Time", key):
                        timestamps.append(value.strip())
                    info_dict[key.strip()] = value.strip()

                assert current_channels is not None
                assert len(timestamps) == 2 * int(info_dict['Number of Seizures in File'])
                info_dict['timestamp'] = timestamps
                info_dict['Channels'] = current_channels
                extracted_info[info_dict['File Name']] = info_dict

    return sample_rate, extracted_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--setting", type=str, choices=["Inductive", "Transductive"], required=True)
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

    with open(os.path.join(origin_dir, 'RECORDS-WITH-SEIZURES'), 'r') as fp:
        files_with_seizures = fp.readlines()
        files_with_seizures = [file.strip().split('/')[1] for file in files_with_seizures if '.edf' in file]
        print(files_with_seizures)

    # load data
    user_id = 0
    all_x, all_y, all_u, all_channels = [], [], [], []
    patient_dirs = list(filter(lambda p: 'chb' in p and os.path.isdir(os.path.join(origin_dir, p)), os.listdir(origin_dir)))
    for patient_dir in patient_dirs:
        print("*" * 30 + patient_dir + "*" * 30)
        user_id = str(patient_dir[3:])
        _all_x, _all_y = [], []
        patient_dir = os.path.join(origin_dir, patient_dir)
        _, extracted_info = load_summary(os.path.join(patient_dir, f"chb{user_id}-summary.txt"))

        skip_files = []
        edf_files = list(filter(lambda f: os.path.splitext(f)[1] == '.edf', os.listdir(patient_dir)))
        for edf_file in edf_files:
            try:
                x = load_edf_data(os.path.join(patient_dir, edf_file), sample_rate)

                if edf_file in extracted_info:
                    y = load_truth_data(extracted_info[edf_file], length=x.shape[0], sample_rate=sample_rate)
                else:
                    y = np.zeros([x.shape[0]], dtype=float)

                if edf_file in files_with_seizures:
                    assert y.any(), f"File {edf_file} should be a seizure file, but found no seizures"
                else:
                    assert (~y.astype(bool)).all(), f"File {edf_file} should be a non-seizure file, but found seizures"

            except Exception as e:
                skip_files.append(edf_file)
                print(f"{edf_file}: Error {repr(e)}")

            else:
                x = x[:len(x) // (window * sample_rate) * (window * sample_rate)]
                y = y[:len(x) // (window * sample_rate) * (window * sample_rate)]

                # drop files that do not have seizures
                if drop_non_seizure_files and (~y.astype(bool)).all():
                    print(f"{edf_file}: Drop since it has no seizures")
                    continue

                print(f"{edf_file}: Loaded")

                _all_x.append(x)
                _all_y.append(y)

        all_u.append(int(user_id))
        all_x.append(np.concatenate(_all_x, axis=0))
        all_y.append(np.concatenate(_all_y, axis=0))

    dest_dir = dest_dir + '-' + setting
    process(all_u, all_x, all_y, sample_rate, window, horizon, stride, patch_len, setting, ratio, dest_dir, split, channels, n_sample_per_file)
