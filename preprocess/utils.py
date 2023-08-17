import numpy as np
from scipy.fftpack import fft
from tqdm import tqdm


def compute_FFT(signals, n):
    """ from git@github.com:tsy935/eeg-gnn-ssl.git
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)

    return FT


def slice_samples(idx, x, label, window, horizon, stride):
    split_u, split_x, split_y, split_label, split_ylabel = [], [], [], [], []
    for i in tqdm(range(len(idx)), desc="Slice samples"):
        u = idx[i]
        assert len(x[i] == len(label[i]))
        _split_x, _split_y, _split_label, _split_ylabel = [], [], [], []
        for j in range(0, len(x[i]) - horizon - window, stride):
            _x = x[i][j:j + window, :]
            _y = x[i][j + window:j + window + horizon, :]
            _l = label[i][j:j + window].astype(bool)
            _yl = label[i][j + window:j + window + horizon].astype(bool)

            _split_x.append(_x)
            _split_y.append(_y)
            _split_label.append(_l)
            _split_ylabel.append(_yl)

        _split_u = np.empty(len(_split_label), dtype=int)
        _split_u.fill(u)
        split_u.append(_split_u)
        split_x.append(np.stack(_split_x, axis=0))
        split_y.append(np.stack(_split_y, axis=0))
        split_label.append(np.stack(_split_label, axis=0))
        split_ylabel.append(np.array(_split_ylabel))

    return split_u, split_x, split_y, split_label, split_ylabel


def segmentation(all_x, seg):
    """
    :param all_x: shape (T, C)[]
    :param seg: int, stands for `D`
    :return: shape (T//D, D, C)[]
    """
    new_x = []
    for x in tqdm(all_x, desc="Segment"):
        segments = []
        for _x in x:
            assert len(_x) % seg == 0
            _segments = []
            for i in range(0, len(_x), seg):
                segment = _x[i:i + seg]  # (D, C)
                _segments.append(segment)
            _segments = np.stack(_segments, axis=0)
            segments.append(_segments)
        x = np.stack(segments, axis=0)  # (T//D, D, C)[]
        new_x.append(x)
    return new_x


def calculate_scaler(x, mode, ratio):
    values, n_samples = [], []
    if mode == 'Transductive':
        for u in range(len(x)):
            train_idx = int(len(x[u]) * ratio[0])
            values.append(x[u][:train_idx])
            n_samples.append(x[u][:train_idx].size)

    elif mode == 'Inductive':
        train_idx = int(len(x) * ratio[0])
        for u in range(len(x))[:train_idx]:
            values.append(x[u])
            n_samples.append(x[u].size)

    # mean
    all_sum = 0
    for v, n in zip(values, n_samples):
        all_sum += v.sum()
    mean = all_sum / sum(n_samples)

    # std
    all_sqrt = 0
    for v, n in zip(values, n_samples):
        all_sqrt += ((v - mean) ** 2).sum()
    std = np.sqrt(all_sqrt / sum(n_samples))

    return mean, std


def calculate_fft_scaler(all_x, mode, ratio, seg):
    fft_x = []
    for x in all_x:
        segments = []
        for _x in x:
            _segments = []
            for segment in _x:  # (D, C)
                segment = compute_FFT(segment.T, n=seg).T
                _segments.append(segment)
            _segments = np.stack(_segments, axis=0)
            segments.append(_segments)
        x = np.stack(segments, axis=0)  # (N, T//D, D, C)
        fft_x.append(x)

    return fft_x, calculate_scaler(fft_x, mode, ratio)


def split_dataset(u, x, y, l, yl, mode, ratio):
    train_u, train_x, train_y, train_l, train_yl = [], [], [], [], []
    val_u, val_x, val_y, val_l, val_yl = [], [], [], [], []
    test_u, test_x, test_y, test_l, test_yl = [], [], [], [], []

    assert len(x) == len(y) == len(l)
    if mode == 'Transductive':
        for i in tqdm(range(len(x)), desc="Split dataset"):
            assert len(u[i]) == len(x[i]) == len(y[i]) == len(l[i]) == len(yl[i])
            n_samples = len(x[i])
            train_idx = int(n_samples * ratio[0])
            val_idx = train_idx + int(n_samples * ratio[1])

            train_u.extend(u[i][:train_idx])
            train_x.extend(x[i][:train_idx])
            train_y.extend(y[i][:train_idx])
            train_l.extend(l[i][:train_idx])
            train_yl.extend(yl[i][:train_idx])

            val_u.extend(u[i][train_idx:val_idx])
            val_x.extend(x[i][train_idx:val_idx])
            val_y.extend(y[i][train_idx:val_idx])
            val_l.extend(l[i][train_idx:val_idx])
            val_yl.extend(yl[i][train_idx:val_idx])

            test_u.extend(u[i][val_idx:])
            test_x.extend(x[i][val_idx:])
            test_y.extend(y[i][val_idx:])
            test_l.extend(l[i][val_idx:])
            test_yl.extend(yl[i][val_idx:])

    elif mode == 'Inductive':
        train_idx = int(len(x) * ratio[0])
        val_idx = train_idx + int(len(x) * ratio[1])
        for i in range(len(x))[:train_idx]:
            train_u.extend(u[i])
            train_x.extend(x[i])
            train_y.extend(y[i])
            train_l.extend(l[i])
            train_yl.extend(yl[i])

        for i in range(len(x))[train_idx:val_idx]:
            val_u.extend(u[i])
            val_x.extend(x[i])
            val_y.extend(y[i])
            val_l.extend(l[i])
            val_yl.extend(yl[i])

        for i in range(len(x))[val_idx:]:
            test_u.extend(u[i])
            test_x.extend(x[i])
            test_y.extend(y[i])
            test_l.extend(l[i])
            test_yl.extend(yl[i])

    else:
        raise ValueError(f"Not implemented mode: {mode}")

    return (train_u, train_x, train_y, train_l, train_yl), \
        (val_u, val_x, val_y, val_l, val_yl), \
        (test_u, test_x, test_y, test_l, test_yl)
