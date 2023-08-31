import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class MinEuclideanDistBlock(nn.Module):
    """
    Calculates the euclidean distances of a bunch of shapelets to a data set and performs global min-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """

    def __init__(self, shapelets_size, num_shapelets, in_channels=1):
        super(MinEuclideanDistBlock, self).__init__()
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True)
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x):
        """
        1) Unfold the data set 2) calculate euclidean distance 3) sum over channels and 4) perform global min-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the euclidean for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2)

        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        # hard min compared to soft-min from the paper
        x, _ = torch.min(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)}"
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)
        self.shapelets.retain_grad()

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)}"
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()
        self.shapelets.retain_grad()


class MaxCosineSimilarityBlock(nn.Module):
    """
    Calculates the cosine similarity of a bunch of shapelets to a data set and performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """

    def __init__(self, shapelets_size, num_shapelets, in_channels=1):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x):
        """
        1) Unfold the data set 2) calculate norm of the data and the shapelets 3) calculate pair-wise dot-product
        4) sum over channels 5) perform a ReLU to ignore the negative values and 6) perform global max-pooling
        @param x: the time series data
        @type x: tensor(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the cosine similarity for each pair of shapelet and time series instance
        @rtype: tensor(num_samples, num_shapelets)
        """
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims
        # ignore negative distances
        x = self.relu(x)
        x, _ = torch.max(x, 3)
        return x

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.transpose(1, 0)

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        # transpose since internally we need shape (in_channels, num_shapelets, shapelets_size)
        weights = weights.transpose(1, 0)

        if not list(weights.shape) == list(self.shapelets.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets.shape)} "
                             f"compared to {list(weights.shape)}")

        self.shapelets = nn.Parameter(weights)

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets[:, j].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape {list(self.shapelets[:, j].shape)} "
                             f"compared to {list(weights[j].shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights, dtype=torch.float)
        self.shapelets[:, j] = weights
        self.shapelets = nn.Parameter(self.shapelets).contiguous()


class MaxCrossCorrelationBlock(nn.Module):
    """
    Calculates the cross-correlation of a bunch of shapelets to a data set, implemented via convolution and
    performs global max-pooling.
    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    cuda : bool
        if true loads everything to the GPU
    """

    def __init__(self, shapelets_size, num_shapelets, in_channels=1):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size

    def forward(self, x):
        """
        1) Apply 1D convolution 2) Apply global max-pooling
        @param x: the data set of time series
        @type x: array(float) of shape (num_samples, in_channels, len_ts)
        @return: Return the most similar values for each pair of shapelet and time series instance
        @rtype: tensor(n_samples, num_shapelets)
        """
        x = self.shapelets(x)
        x, _ = torch.max(x, 2, keepdim=True)
        return x.transpose(2, 1)

    def get_shapelets(self):
        """
        Return the shapelets contained in this block.
        @return: An array containing the shapelets
        @rtype: tensor(float) with shape (num_shapelets, in_channels, shapelets_size)
        """
        return self.shapelets.weight.data

    def set_shapelet_weights(self, weights):
        """
        Set weights for all shapelets in this block.
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (num_shapelets, in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)

        if not list(weights.shape) == list(self.shapelets.weight.data.shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data.shape)} compared to {list(weights.shape)}")

        self.shapelets.weight.data = weights

    def set_weights_of_single_shapelet(self, j, weights):
        """
        Set the weights of a single shapelet.
        @param j: The index of the shapelet to set
        @type j: int
        @param weights: the weights for the shapelet
        @type weights: array-like(float) of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        if not list(weights.shape) == list(self.shapelets.weight.data[j, :].shape):
            raise ValueError(f"Shapes do not match. Currently set weights have shape"
                             f"{list(self.shapelets.weight.data[j, :].shape)} compared to {list(weights.shape)}")
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        self.shapelets.weight.data[j, :] = weights


class ShapeletsDistBlocks(nn.Module):
    """
    Defines a number of blocks containing a number of shapelets, whereas
    the shapelets in each block have the same size.
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        keys are the length of the shapelets for a block and the values the number of shapelets for the block
    in_channels : int
        the number of input channels of the dataset
    dist_measure: 'string'
        the distance measure, either of 'euclidean', 'cross-correlation', or 'cosine'
    """

    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean'):
        super(ShapeletsDistBlocks, self).__init__()
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def forward(self, x):
        """
        Calculate the distances of each shapelet block to the time series data x and concatenate the results.
        @param x: the time series data
        @type x: tensor(float) of shape (n_samples, in_channels, len_ts)
        @return: a distance matrix containing the distances of each shapelet to the time series data
        @rtype: tensor(float) of shape
        """
        out = torch.tensor([], dtype=torch.float).to(x.device)
        for block in self.blocks:
            out = torch.cat((out, block(x)), dim=2)

        return out

    def get_blocks(self):
        """
        @return: the list of shapelet blocks
        @rtype: nn.ModuleList
        """
        return self.blocks

    def get_block(self, i):
        """
        Get a specific shapelet block. The blocks are ordered (ascending) according to the shapelet lengths.
        @param i: the index of the block to fetch
        @type i: int
        @return: return shapelet block i
        @rtype: nn.Module, either
        """
        return self.blocks[i]

    def set_shapelet_weights_of_block(self, i, weights):
        """
        Set the weights of the shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param weights: the weights to set for the shapelets
        @type weights: array-like(float) of shape (in_channels, num_shapelets, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_shapelet_weights(weights)

    def get_shapelets_of_block(self, i):
        """
        Return the shapelet of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @return: the weights of the shapelet block
        @rtype: tensor(float) of shape (in_channels, num_shapelets, shapelets_size)
        """
        return self.blocks[i].get_shapelets()

    def get_shapelet(self, i, j):
        """
        Return the shapelet at index j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @return: return the weights of the shapelet
        @rtype: tensor(float) of shape
        """
        shapelet_weights = self.blocks[i].get_shapelets()
        return shapelet_weights[j, :]

    def set_shapelet_weights_of_single_shapelet(self, i, j, weights):
        """
        Set the weights of shapelet j of shapelet block i.
        @param i: the index of the shapelet block
        @type i: int
        @param j: the index of the shapelet in shapelet block i
        @type j: int
        @param weights: the new weights for the shapelet
        @type weights: array-like of shape (in_channels, shapelets_size)
        @return:
        @rtype: None
        """
        self.blocks[i].set_weights_of_single_shapelet(j, weights)

    def get_shapelets(self):
        """
        Return a matrix of all shapelets. The shapelets are ordered (ascending) according to
        the shapelet lengths and padded with NaN.
        @return: a tensor of all shapelets
        @rtype: tensor(float) with shape (in_channels, num_total_shapelets, shapelets_size_max)
        """
        max_shapelet_len = max(self.shapelets_size_and_len.keys())
        num_total_shapelets = sum(self.shapelets_size_and_len.values())
        shapelets = torch.Tensor(num_total_shapelets, self.in_channels, max_shapelet_len)
        shapelets[:] = np.nan
        start = 0
        for block in self.blocks:
            shapelets_block = block.get_shapelets()
            end = start + block.num_shapelets
            shapelets[start:end, :, :block.shapelets_size] = shapelets_block
            start += block.num_shapelets
        return shapelets
