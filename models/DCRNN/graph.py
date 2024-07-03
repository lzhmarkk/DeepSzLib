import torch
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


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


def calculate_normalized_laplacian_pytorch(adj):
    device = adj.device
    d = adj.sum(dim=-1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    identity = torch.eye(adj.size(1), device=device)
    normalized_laplacian = identity - torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
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


def calculate_scaled_laplacian_pytorch(adj_mx, lambda_max=2):
    # (bs, num_nodes, num_nodes)
    assert isinstance(adj_mx, torch.Tensor) and adj_mx.ndim == 3
    adj_mx = torch.maximum(adj_mx, adj_mx.transpose(1, 2))
    L = calculate_normalized_laplacian_pytorch(adj_mx)
    I = torch.eye(adj_mx.shape[1]).to(adj_mx.device)
    L = (2 / lambda_max * L) - I
    return L


def norm_graph(adj_mat, filter_type):
    supports = []
    supports_mat = []
    if isinstance(adj_mat, torch.Tensor):
        supports.append(calculate_scaled_laplacian_pytorch(adj_mat))

    elif isinstance(adj_mat, np.ndarray):
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


def distance_support(channels):
    with open(f"./data/electrode_graph/adj_mx_3d.pkl", 'rb') as pf:
        adj_mat = pickle.load(pf)
        adj_mat = adj_mat[-1]
        INCLUDED_CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']

    # mapping
    channel_mapping = [[], []]
    for i, c in enumerate(channels):
        for j, ref in enumerate(INCLUDED_CHANNELS):
            if ref.lower() in c.lower():
                channel_mapping[0].append(i)
                channel_mapping[1].append(j)
                break

    print(f"Channel mapping {channel_mapping}")
    channel_mapping = np.array(channel_mapping).astype(int)
    new_adj_mat = np.eye(len(channels))
    for i, j in zip(*channel_mapping):
        new_adj_mat[i, channel_mapping[0]] = adj_mat[j, channel_mapping[1]]

    return new_adj_mat


def correlate_pytorch(in1, in2, mode='full'):
    assert in1.ndim == 2 and in2.ndim == 1
    bs = in1.shape[0]
    in1 = in1.view(bs, 1, in1.shape[1])
    in2 = in2.view(1, 1, in2.shape[0])

    if mode == 'full':
        output = torch.nn.functional.conv1d(in1, in2.flip(2), padding=in2.size(-1) - 1)
    elif mode == 'same':
        output = torch.nn.functional.conv1d(in1, in2.flip(2), padding=in2.size(-1) // 2)
    elif mode == 'valid':
        output = torch.nn.functional.conv1d(in1, in2.flip(2))
    else:
        raise ValueError("Invalid mode: must be 'full', 'same', or 'valid'.")

    return output.view(-1)


def comp_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 2D array
        y: 1D array
        mode: 'valid', 'full' or 'same',
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        normalize: If True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """
    xcorr = correlate_pytorch(x, y, mode=mode)
    # the below normalization code refers to matlab xcorr function
    cxx0 = (torch.abs(x) ** 2).sum(dim=-1)
    cyy0 = (torch.abs(y) ** 2).sum()
    if normalize and (cxx0 != 0).any() and (cyy0 != 0):
        scale = (cxx0 * cyy0) ** 0.5
        xcorr[cxx0 != 0] /= scale
    return xcorr


def correlation_support(eeg_clip):
    """
    Compute adjacency matrix for correlation graph
    Args:
        eeg_clip: shape (bs, seq_len, num_nodes, input_dim)
    Returns:
        adj_mat: adjacency matrix, shape (bs, num_nodes, num_nodes)
    """
    bs = eeg_clip.shape[0]
    num_sensors = eeg_clip.shape[2]

    # (bs, num_nodes, seq_len, input_dim)
    eeg_clip = eeg_clip.transpose(2, 1)
    assert eeg_clip.shape[1] == num_sensors

    # (num_nodes, seq_len*input_dim)
    eeg_clip = eeg_clip.reshape((bs, num_sensors, -1))

    adj_mat_batched = []
    for bid in range(bs):
        adj_mat = torch.eye(num_sensors, num_sensors).to(eeg_clip.device)  # diagonal is 1
        for j in range(1, num_sensors):
            xcorr = comp_xcorr(eeg_clip[bid, :j, :], eeg_clip[bid, j, :], mode='valid', normalize=True)
            adj_mat[:j, j] = xcorr
            adj_mat[j, :j] = xcorr
        # ignore topk operation
        adj_mat = torch.abs(adj_mat)
        adj_mat_batched.append(adj_mat)

    return torch.stack(adj_mat_batched, dim=0)
