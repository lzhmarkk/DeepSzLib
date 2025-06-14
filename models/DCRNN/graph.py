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


def correlation_support(eeg_clip):
    """
    计算相关图的邻接矩阵 (GPU 优化版)
    Generated by Gemini2.5-pro-preview-0605

    Args:
        eeg_clip: shape (bs, seq_len, num_nodes, input_dim)

    Returns:
        adj_mat: 邻接矩阵, shape (bs, num_nodes, num_nodes)
    """
    # 1. 重塑输入张量
    # 原始形状: (bs, seq_len, num_nodes, input_dim)
    bs, _, num_nodes, _ = eeg_clip.shape

    # 交换维度并将 seq_len 和 input_dim 合并
    # 新形状: (bs, num_nodes, seq_len * input_dim)
    # 我们称这个合并后的维度为 feature_len
    eeg_clip_reshaped = eeg_clip.transpose(1, 2).reshape(bs, num_nodes, -1)

    # 2. 以批处理方式计算成对的点积
    # 这是余弦相似度的分子部分。
    # (bs, num_nodes, feature_len) @ (bs, feature_len, num_nodes) -> (bs, num_nodes, num_nodes)
    dot_product = torch.bmm(eeg_clip_reshaped, eeg_clip_reshaped.transpose(1, 2))

    # 3. 计算每个传感器信号的 L2 范数
    # 这是余弦相似度的分母部分。
    # 输出形状: (bs, num_nodes)
    norms = torch.norm(eeg_clip_reshaped, p=2, dim=-1)

    # 4. 计算范数的外积以获得缩放矩阵
    # (bs, num_nodes, 1) @ (bs, 1, num_nodes) -> (bs, num_nodes, num_nodes)
    # scale_matrix[b, i, j] = norms[b, i] * norms[b, j]
    scale_matrix = torch.bmm(norms.unsqueeze(2), norms.unsqueeze(1))

    # 5. 计算最终的相关矩阵 (即余弦相似度)
    # 添加一个很小的 epsilon 以保证数值稳定性，避免除以零。
    epsilon = 1e-9
    adj_mat = dot_product / (scale_matrix + epsilon)

    # 6. 像原始函数一样取绝对值
    adj_mat = torch.abs(adj_mat)

    # 对角线上的元素是信号与自身的相关性，根据上面的计算，它已经是 1。
    # 这与原始代码中用 torch.eye() 将对角线设为 1 的效果一致。

    return adj_mat
