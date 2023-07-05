import torch
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.signal import correlate


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    State transition matrix D_o^-1W in paper.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    """
    Reverse state transition matrix D_i^-1W^T in paper.
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


def norm_graph(adj_mat, filter_type):
    supports = []
    supports_mat = []
    if filter_type == "laplacian":  # ChebNet graph conv
        supports_mat.append(calculate_scaled_laplacian(adj_mat))
    elif filter_type == "random_walk":  # Forward random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
    elif filter_type == "dual_random_walk":  # Bidirectional random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(calculate_scaled_laplacian(adj_mat))
    for support in supports_mat:
        supports.append(torch.FloatTensor(support.toarray()))
    return supports


def distance_support(n_channels):
    with open(f"./data/electrode_graph/adj_mx_3d.pkl", 'rb') as pf:
        adj_mat = pickle.load(pf)
        adj_mat = adj_mat[-1]

    # mapping
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'T3', 'T4', 'EKG', 'EMG']
    INCLUDED_CHANNELS = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
                         'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG FZ', 'EEG CZ', 'EEG PZ']
    channel_mapping = []
    for c in channels:
        for i, ref in enumerate(INCLUDED_CHANNELS):
            if str(c).upper() in ref:
                channel_mapping.append(i)
                break
    print(f"Channel mapping {channel_mapping}")
    channel_mapping = np.array(channel_mapping).astype(int)
    new_adj_mat = np.eye(n_channels)
    new_adj_mat[:len(channel_mapping), :len(channel_mapping)] = adj_mat[channel_mapping][:, channel_mapping]
    adj_mat = new_adj_mat

    return adj_mat


def comp_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 1D array
        y: 1D array
        mode: 'valid', 'full' or 'same',
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        normalize: If True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """
    xcorr = correlate(x, y, mode=mode)
    # the below normalization code refers to matlab xcorr function
    cxx0 = (np.absolute(x) ** 2).sum()
    cyy0 = (np.absolute(y) ** 2).sum()
    if normalize and (cxx0 != 0) and (cyy0 != 0):
        scale = (cxx0 * cyy0) ** 0.5
        xcorr /= scale
    return xcorr


def correlation_support(eeg_clip):
    """
    Compute adjacency matrix for correlation graph
    Args:
        eeg_clip: shape (seq_len, num_nodes, input_dim)
    Returns:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
    """
    num_sensors = eeg_clip.shape[1]
    adj_mat = np.eye(num_sensors, num_sensors, dtype=np.float32)  # diagonal is 1

    # (num_nodes, seq_len, input_dim)
    eeg_clip = np.transpose(eeg_clip, (1, 0, 2))
    assert eeg_clip.shape[0] == num_sensors

    # (num_nodes, seq_len*input_dim)
    eeg_clip = eeg_clip.reshape((num_sensors, -1))

    for i in range(0, num_sensors):
        for j in range(i + 1, num_sensors):
            xcorr = comp_xcorr(
                eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
            adj_mat[i, j] = xcorr
            adj_mat[j, i] = xcorr

    adj_mat = abs(adj_mat)
    # ignore topk operation
    return adj_mat
