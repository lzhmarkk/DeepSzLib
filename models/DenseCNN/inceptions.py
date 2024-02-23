import torch
import torch.nn as nn
from .conv import BasicConv2d


class Inception(nn.Module):
    def __init__(self, in_channels, pool_features, filter_size, pool_size):
        super(Inception, self).__init__()
        self.filter_size = filter_size
        self.pool_size = pool_size

        self.branchA_1 = BasicConv2d(in_channels, pool_features, False, kernel_size=(filter_size[0], 1), stride=1,
                                     padding=(int((filter_size[0] - 1) / 2), 0))

        self.branchA_2 = BasicConv2d(pool_features, pool_features, False, kernel_size=(filter_size[0], 1), stride=1,
                                     padding=(int((filter_size[0] - 1) / 2), 0))

        self.branchB_1 = BasicConv2d(in_channels, pool_features, False, kernel_size=(filter_size[1], 1), stride=1,
                                     padding=(int((filter_size[1] - 1) / 2), 0))

        self.branchB_2 = BasicConv2d(pool_features, pool_features, False, kernel_size=(filter_size[1], 1), stride=1,
                                     padding=(int((filter_size[1] - 1) / 2), 0))

        self.branchB_3 = BasicConv2d(pool_features, pool_features, False, kernel_size=(filter_size[1], 1), stride=1,
                                     padding=(int((filter_size[1] - 1) / 2), 0))

        self.branchC_1 = BasicConv2d(in_channels, pool_features, False, kernel_size=(filter_size[2], 1), stride=1,
                                     padding=(int((filter_size[2] - 1) / 2), 0))
        self.branchC_1 = BasicConv2d(in_channels, pool_features, False, kernel_size=(21, 1), stride=1, padding=(10, 0))
        self.branchC_2 = BasicConv2d(pool_features, pool_features, False, kernel_size=(21, 1), stride=1,
                                     padding=(10, 0))
        self.branchC_2 = BasicConv2d(pool_features, pool_features, False, kernel_size=(filter_size[2], 1), stride=1,
                                     padding=(int((filter_size[2] - 1) / 2), 0))

    def forward(self, x):
        branchA_1 = self.branchA_1(x)
        branchA_2 = self.branchA_2(branchA_1)

        branchB_1 = self.branchB_1(x)
        branchB_2 = self.branchB_2(branchB_1)
        branchB_3 = self.branchB_3(branchB_2)

        branchC_1 = self.branchC_1(x)
        branchC_2 = self.branchC_2(branchC_1)

        outputs = [branchA_2, branchB_3, branchC_2]
        return torch.cat(outputs, 1)
