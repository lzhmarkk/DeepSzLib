import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.stats import entropy


def mean(seq):
    return np.mean(seq, axis=0)


def variance(seq):
    return np.var(seq, axis=0)


def standard_deviation(seq):
    return np.std(seq, axis=0)


def mode(seq):
    modes = stats.mode(seq, axis=0, keepdims=True)
    return modes.mode[0]


def median(seq):
    return np.median(seq, axis=0)


def skewness(seq):
    return stats.skew(seq, axis=0)


def kurtosis(seq):
    return stats.kurtosis(seq, axis=0)


def maximum(seq):
    return np.max(seq, axis=0)


def minimum(seq):
    return np.min(seq, axis=0)


def line_length(seq):
    return np.sum(np.abs(np.diff(seq, axis=0)), axis=0)


def energy(seq):
    return np.sum(np.square(seq), axis=0)


def power(seq):
    return np.mean(np.square(seq), axis=0)


def shannon_entropy(seq):
    T, D = seq.shape
    entropy = np.zeros(D)

    for d in range(D):
        _, counts = np.unique(seq[:, d], return_counts=True)
        probs = counts / T
        entropy[d] = -np.sum(probs * np.log2(probs))

    return entropy


def spectral_power(spe):
    power_spectrum = np.square(np.abs(spe))
    power_spectrum /= spe.shape[1]
    return power_spectrum


def spectral_entropy(spe):
    entropy_values = np.zeros(spe.shape[0])
    for b in range(spe.shape[0]):
        power_spectrum = np.square(np.abs(spe[b]))
        power_spectrum /= spe.shape[1]
        prob_distribution = power_spectrum / np.sum(power_spectrum)
        entropy_values[b] = entropy(prob_distribution)
    return entropy_values


def frequency_energy(spe):
    return np.sum(np.square(np.abs(spe)), axis=1)


def peak_frequency(spe):
    freqs = fftfreq(spe.shape[1])
    max_indices = np.argmax(np.abs(spe), axis=1)
    peak_freqs = np.abs(freqs[max_indices])
    return peak_freqs


def median_frequency(spe):
    freqs = fftfreq(spe.shape[1])
    cumulative_sum = np.cumsum(np.abs(spe), axis=1)
    median_indices = np.searchsorted(cumulative_sum, cumulative_sum[:, -1, None] / 2, axis=1)
    median_freqs = np.abs(freqs[median_indices])
    return median_freqs


def extract_features(sequences, axis, preprocess):
    features = []
    assert axis == 1

    for seq in sequences:
        if preprocess == 'seg':
            feature = [mean(seq), variance(seq), standard_deviation(seq), median(seq), skewness(seq),
                       kurtosis(seq), maximum(seq), minimum(seq), line_length(seq), energy(seq),
                       power(seq), mode(seq), shannon_entropy(seq)]
            feature = np.stack(feature, axis=-1)
            features.append(feature)
        elif preprocess == 'fft':
            feature = [spectral_power(seq), spectral_entropy(seq), frequency_energy(seq), peak_frequency(seq),
                       median_frequency(seq)]
            feature = np.stack(feature, axis=-1)
            features.append(feature)

    features = np.stack(features, axis=0)
    features[np.isnan(features)] = 0.
    return features
