import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import BasicConv2d
from .inceptions import Inception
from models.utils import check_tasks


class DenseCNN(nn.Module):
    supported_tasks = ['detection', 'onset_detection', 'classification']
    unsupported_tasks = ['prediction']

    def __init__(self, args):
        super().__init__()
        self.num_channels = args.hidden
        self.n_nodes = args.n_channels
        self.num_inception_layers = args.num_inception_layers
        self.window = args.window
        self.patch_len = args.patch_len
        self.preprocess = args.preprocess
        self.input_dim = args.input_dim
        self.seq_len = self.window // self.patch_len * self.input_dim
        self.onset_history_len = args.onset_history_len
        self.task = args.task
        check_tasks(self)

        self.inception_0 = Inception(1, pool_features=self.num_channels, filter_size=[9, 15, 21], pool_size=5)
        self.inception_1 = Inception(self.num_channels * 3, pool_features=self.num_channels * 3,
                                     filter_size=[9, 13, 17], pool_size=4)
        self.conv1x1_10 = BasicConv2d(self.num_channels * 12, self.num_channels * 9, False, kernel_size=(1, 1),
                                      stride=1)

        self.inception_2 = Inception(self.num_channels * 9, pool_features=self.num_channels * 9,
                                     filter_size=[7, 11, 15], pool_size=4)

        self.conv1x1_2 = BasicConv2d(self.num_channels * 27, self.num_channels * 18, False, kernel_size=(1, 1),
                                     stride=1)

        self.inception_3 = Inception(self.num_channels * 18, pool_features=self.num_channels * 18,
                                     filter_size=[5, 7, 9], pool_size=3)

        self.conv1x1_3 = BasicConv2d(self.num_channels * 54, self.num_channels * 18, False, kernel_size=(1, 1),
                                     stride=1)

        self.conv1x1_32 = BasicConv2d(self.num_channels * 36, self.num_channels * 18, False, kernel_size=(1, 1),
                                      stride=1)

        self.inception_4 = Inception(self.num_channels * 18, pool_features=self.num_channels * 18,
                                     filter_size=[3, 5, 7], pool_size=3)

        self.conv1x1_4 = BasicConv2d(self.num_channels * 54, self.num_channels * 18, False, kernel_size=(1, 1),
                                     stride=1)

        self.conv1x1_5 = BasicConv2d(self.num_channels * 54, self.num_channels * 27, False, kernel_size=(1, 1),
                                     stride=1)

        self.conv1x1_54 = BasicConv2d(self.num_channels * 45, self.num_channels * 18, False, kernel_size=(1, 1),
                                      stride=1)

        self.inception_6 = Inception(self.num_channels * 18, pool_features=self.num_channels * 18,
                                     filter_size=[3, 5, 7], pool_size=3)

        self.conv1x1_6 = BasicConv2d(self.num_channels * 54, self.num_channels * 18, False, kernel_size=(1, 1),
                                     stride=1)

        self.conv1x1_7 = BasicConv2d(self.num_channels * 54, self.num_channels * 27, False, kernel_size=(1, 1),
                                     stride=1)

        self.conv1x1_76 = BasicConv2d(self.num_channels * 45, self.num_channels * 36, False, kernel_size=(1, 1),
                                      stride=1)

        if 'onset_detection' in self.task:
            self.fc1 = nn.Linear(self.num_channels * 36 * int(self.onset_history_len / (7 * 5 * 5 * 4)), 32)
        elif 'detection' in self.task or 'classification' in self.task:
            self.fc1 = nn.Linear(self.num_channels * 36 * int(self.seq_len / (7 * 5 * 5 * 4)), 32)
        else:
            raise NotImplementedError
        self.fcbn1 = nn.BatchNorm1d(32)
        self.dropout_rate = args.dropout

        self.fc2 = nn.Linear(32 * self.n_nodes, args.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def predict(self, x):
        bs = x.shape[0]
        x = x.transpose(3, 2).reshape(bs, self.seq_len, self.n_nodes)
        s = x.unsqueeze(1)  # B x 1 x T x C

        s_0 = self.inception_0(s)
        s_1 = self.inception_1(s_0)
        s_cat_10 = torch.cat([s_0, s_1], 1)
        s = self.conv1x1_10(s_cat_10)
        s = F.max_pool2d(s, (7, 1))

        s_0 = self.inception_2(s)
        s_0 = self.conv1x1_2(s_0)
        s_1 = self.inception_3(s_0)
        s_1 = self.conv1x1_3(s_1)
        s_cat_10 = torch.cat([s_0, s_1], 1)
        s = self.conv1x1_32(s_cat_10)
        s = F.max_pool2d(s, (5, 1))

        s_0 = self.inception_4(s)
        s_0 = self.conv1x1_4(s_0)
        s_1 = self.inception_4(s_0)
        s_1 = self.conv1x1_5(s_1)
        s_cat_10 = torch.cat([s_0, s_1], 1)
        s = self.conv1x1_54(s_cat_10)
        s = F.max_pool2d(s, (5, 1))

        s_0 = self.inception_6(s)
        s_0 = self.conv1x1_6(s_0)
        s_1 = self.inception_6(s_0)
        s_1 = self.conv1x1_7(s_1)
        s_cat_10 = torch.cat([s_0, s_1], 1)
        s = self.conv1x1_76(s_cat_10)
        s = F.max_pool2d(s, (4, 1))

        s = s.contiguous()
        s = s.reshape(bs, -1, self.n_nodes).transpose(2, 1)
        s = self.fc1(s).transpose(2, 1)
        s = F.dropout(F.relu(self.fcbn1(s)), p=self.dropout_rate, training=self.training)

        z = self.fc2(s.reshape(bs, -1)).squeeze(dim=-1)
        return z

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        if 'onset_detection' in self.task:
            out = []
            for t in range(1, self.window // self.patch_len + 1):
                xt = x[:, max(0, t - self.onset_history_len):t, :, :]
                if xt.shape[-3] < self.onset_history_len:
                    xt = nn.functional.pad(xt, (0, 0, 0, 0, self.onset_history_len - xt.shape[-3], 0))

                zt = self.predict(xt)
                out.append(zt)
            z = torch.stack(out, dim=1)
        elif 'detection' in self.task or 'classification' in self.task:
            z = self.predict(x)
        else:
            raise NotImplementedError

        return {'prob': z}
