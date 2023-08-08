import torch
import numpy as np


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output
