import torch
import random
import numpy as np
from torch import nn
from .blocks import ShapeletsDistBlocks
from tslearn.clustering import TimeSeriesKMeans
from collections import defaultdict


class Shapelet(nn.Module):
    """
    Implements Learning Shapelets. Just puts together the ShapeletsDistBlocks with a
    linear layer on top.
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    num_classes: int
        the number of classes for classification
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    """

    def __init__(self, args):
        super(Shapelet, self).__init__()

        self.shapelets_size_and_len = {int(k): int(v) for k, v in args.shapelets_size_and_len.items()}
        self.in_channels = args.n_channels
        self.dist_measure = args.dist_measure
        self.seq_len = args.window
        self.preprocess = args.preprocess
        assert self.preprocess == 'seg'

        self.num_shapelets = sum(self.shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=self.in_channels,
                                                    shapelets_size_and_len=self.shapelets_size_and_len,
                                                    dist_measure=self.dist_measure)
        self.ln = nn.LayerNorm(self.num_shapelets)

        self.task = args.task
        assert 'pred' not in self.task
        assert 'cls' in self.task or 'anomaly' in self.task
        if 'cls' in self.task:
            self.linear = nn.Linear(self.num_shapelets, 1)
        else:
            self.linear = nn.Linear(self.num_shapelets, self.seq_len // args.seg)

        train_set = args.data['train']
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            weights_block = self.get_weights_via_kmeans(train_set, train_set.n_samples_per_file, shapelets_size, num_shapelets)
            self.set_shapelet_weights_of_block(i, weights_block)
            print(f"Initialize shapelet. size: {shapelets_size}, num: {num_shapelets}")

    def forward(self, x, p, y):
        """
        Calculate the distances of each time series to the shapelets and stack a linear layer on top.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, len_ts//len_seg, in_channels, len_seg)
        @return: the logits for the class predictions of the model
        @rtype: tensor(float) of shape (num_samples)
        """
        x = x.transpose(2, 1).reshape(x.shape[0], self.in_channels, self.seq_len)

        x = self.shapelets_blocks(x).squeeze(dim=1)
        x = self.ln(x)
        x = self.linear(x).squeeze()
        return x, None

    def get_weights_via_kmeans(self, dataset, n_samples_per_file, shapelets_size, num_shapelets, n_segments=5000):
        """
        Get weights via k-Means for a block of shapelets.
        """
        segments = []
        for i in range(n_segments):
            sample_index = random.choice(range(len(dataset)))
            file_index = [sample_index // n_samples_per_file]
            sample = dataset[file_index][1].transpose(0, 1, 3, 2).reshape(self.seq_len, self.in_channels)
            segment_start = random.choice(range(len(sample) - shapelets_size))
            segment = sample[segment_start:segment_start + shapelets_size]
            segments.append(segment)

        segments = np.stack(segments, axis=0)
        k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
        clusters = k_means.cluster_centers_.transpose(0, 2, 1)
        return clusters

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        return self.shapelets_blocks.get_shapelets()

    def set_shapelet_weights(self, weights):
        """
        Set the weights of all shapelets. The shapelet weights are expected to be ordered ascending according to the
        length of the shapelets. The values in the matrix for shapelets of smaller length than the maximum
        length are just ignored.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_total_shapelets, shapelets_size_max)
        @return:
        @rtype: None
        """
        start = 0
        for i, (shapelets_size, num_shapelets) in enumerate(self.shapelets_size_and_len.items()):
            end = start + num_shapelets
            self.set_shapelet_weights_of_block(i, weights[start:end, :, :shapelets_size])
            start = end

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of shapelet block i.
        @param i: The index of the shapelet block
        @type i: int
        @param weights: the weights for the shapelets of block i
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_block(i, weights)

    def set_weights_of_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j in shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.shapelets_blocks.set_shapelet_weights_of_single_shapelet(i, j, weights)
