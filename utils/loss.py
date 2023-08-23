import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean()


class GHMCLoss(nn.Module):
    # https://github.com/libuyu/GHM_Detection/blob/d3e82f3d08d70af442b4295fb1f27151f5ab5a1c/mmdetection/mmdet/core/loss/ghm_loss.py
    def __init__(self, bins=10, momentum=0):
        super().__init__()

        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def forward(self, input, target):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        tot = len(input)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(input, target, weights, reduction='sum') / tot
        return loss


class LDAMLoss(nn.Module):
    # https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, input, target):
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m

        output = torch.where(index, x_m, input)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class MyLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cls_loss = args.cls_loss
        self.multi_task = args.multi_task
        self.lamb = args.lamb
        self.scaler = args.scaler
        self.device = args.device

        if self.cls_loss == 'BCE':
            self.cls_loss_fn = nn.BCEWithLogitsLoss()
        elif self.cls_loss == "WBCE":
            self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.n_train / args.n_pos_train - 1]).to(self.device))
        elif self.cls_loss == 'Focal':
            self.cls_loss_fn = FocalLoss()
        elif self.cls_loss == 'GHMC':
            self.cls_loss_fn = GHMCLoss()
        elif self.cls_loss == 'LDAM':
            self.cls_loss_fn = LDAMLoss([args.n_train - args.n_pos_train, args.n_pos_train])
        else:
            raise ValueError(f"Not implemented classification loss: {self.cls_loss}")

        if self.multi_task:
            if self.pred_loss == 'MAE':
                self.pred_loss_fn = nn.L1Loss()
            elif self.pred_loss == 'MSE':
                self.pred_loss_fn = nn.MSELoss()
            elif self.pred_loss == 'CE':
                self.pred_loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Not implemented prediction loss: {self.pred_loss}")

    def forward(self, z, label, truth):
        """
        :param z: tuple(cls_prob, pred_value) or cls_prob
        :param label: cls_truth
        :param truth: pred_truth
        """
        if self.multi_task:
            p, y = z
        else:
            p, y = z, None

        loss = self.cls_loss_fn(input=p, target=label)

        if self.multi_task:
            y = self.scaler.inv_transform(y)
            truth = self.scaler.inv_transform(truth)
            loss += self.lamb * self.pred_loss_fn(input=y, target=truth)

        return loss
