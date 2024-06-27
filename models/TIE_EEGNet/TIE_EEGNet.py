import torch
import torch.nn as nn
from .layer import TIE_Layer
from models.utils import check_tasks

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class TIE_EEGNet(nn.Module):
    supported_tasks = ['detection', 'classification']
    unsupported_tasks = ['prediction', 'onset_detection']

    def ClassifierBlock(self, inputSize, n_class):
        return nn.Sequential(
            nn.Linear(inputSize, n_class, bias=False)
        )

    def __init__(self, args):
        super(TIE_EEGNet, self).__init__()
        self.F1 = args.hidden // 8
        self.F2 = args.hidden // 4
        self.D = args.D
        self.samples = args.window
        self.window = args.window
        self.horizon = args.horizon
        self.channels = args.n_channels
        self.dropoutRate = args.dropoutRate
        self.kernel_length = args.kernel_length
        self.kernel_length2 = args.kernel_length2
        self.tie = args.tie
        self.isTrain = True
        self.alpha_list = args.alpha_list
        self.alpha = args.alpha
        self.pool = args.pool
        self.preprocess = args.preprocess
        assert self.preprocess == 'raw'

        self.Conv2d_1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), padding=(0, self.kernel_length // 2), bias=False)  # 'same'

        self.TIE_Layer = TIE_Layer(pool=self.pool, alpha_list=self.alpha_list, tie=self.tie, conv2Doutput=self.Conv2d_1, inc=1, outc=self.F1,
                                   kernel_size=(1, self.kernel_length), pad=(0, self.kernel_length // 2), stride=1, bias=False,
                                   sample_len=self.samples, is_Train=self.isTrain, alpha=self.alpha)
        self.BatchNorm_1_1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.Depthwise_Conv2d = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, max_norm=1, groups=self.F1,
                                                     bias=False)  # , padding='valid'

        self.BatchNorm_1_2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        self.avg_pool_1 = nn.AvgPool2d((1, 4), stride=4)
        self.Dropout_1 = nn.Dropout(p=self.dropoutRate)
        self.Separable_Conv2d_1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_length2), padding=(0, self.kernel_length // 2),
                                            bias=False, groups=self.F1 * self.D)  # 'same'
        self.Separable_Conv2d_2 = nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), bias=False, groups=1)
        self.BatchNorm_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.avg_pool_2 = nn.AvgPool2d((1, 8), stride=8)
        self.Dropout_2 = nn.Dropout(p=self.dropoutRate)

        self.fea_model = nn.Sequential(self.TIE_Layer,
                                       self.BatchNorm_1_1,
                                       self.Depthwise_Conv2d,
                                       # Activation
                                       self.BatchNorm_1_2,
                                       nn.ELU(inplace=True),
                                       self.avg_pool_1,
                                       self.Dropout_1,
                                       self.Separable_Conv2d_1,
                                       self.Separable_Conv2d_2,
                                       self.BatchNorm_2,
                                       nn.ELU(inplace=True),
                                       self.avg_pool_2,
                                       self.Dropout_2)

        # self.fea_out_size = self.CalculateOutSize(self.fea_model, self.channels, self.samples)
        self.task = args.task
        check_tasks(self)
        self.classifierBlock = self.ClassifierBlock(self.F2 * 99, args.n_classes)

    def forward(self, x, p, y):
        # (B, T, C, D)
        x = x.transpose(1, 2).reshape(-1, 1, self.channels, self.window)  # (B, 1, C, T')
        conv_data = self.fea_model(x)
        flatten_data = conv_data.view(conv_data.size()[0], -1)
        pred_label = self.classifierBlock(flatten_data).squeeze(dim=-1)  # (B)

        return {'prob': pred_label}
