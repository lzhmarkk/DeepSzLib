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
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list).to(device)
        self.device = device
        self.s = s
        self.weight = weight

    def forward(self, input, target):
        batch_m = self.m_list[target.long()]
        x_m = input - batch_m

        output = torch.where(target.bool(), x_m, input)

        return F.binary_cross_entropy_with_logits(self.s * output, target.float(), weight=self.weight)


class MyLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.task = args.task
        self.detection_loss = args.detection_loss
        self.onset_detection_loss = args.onset_detection_loss
        self.classification_loss = args.classification_loss
        self.prediction_loss = args.prediction_loss
        self.lamb = args.lamb
        self.scaler = args.scaler
        self.device = args.device

        if 'detection' in self.task:
            if self.detection_loss == 'BCE':
                self.detection_loss_fn = nn.BCEWithLogitsLoss()
            elif self.detection_loss == 'BCENoSigmoid':
                self.detection_loss_fn = nn.BCELoss()
            elif self.detection_loss == "WBCE":
                weight = torch.tensor([(args.n_train - args.n_pos_train) / args.n_pos_train])
                self.detection_loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
            elif self.detection_loss == 'Focal':
                self.detection_loss_fn = FocalLoss()
            elif self.detection_loss == 'GHMC':
                self.detection_loss_fn = GHMCLoss()
            elif self.detection_loss == 'LDAM':
                self.detection_loss_fn = LDAMLoss([args.n_train - args.n_pos_train, args.n_pos_train], args.device)
            else:
                raise ValueError(f"Not implemented detection loss: {self.detection_loss}")

        elif 'onset_detection' in self.task:
            if self.onset_detection_loss == 'BCE':
                self.onset_detection_loss_fn = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Not implemented onset detection loss: {self.onset_detection_loss}")

        elif 'classification' in self.task:
            if self.classification_loss == 'CE':
                self.classification_loss_fn = nn.CrossEntropyLoss()
            elif self.classification_loss == 'WCE':
                weight = torch.tensor(args.class_count)
                weight = weight.sum() / weight
                self.classification_loss_fn = nn.CrossEntropyLoss(weight=weight)
            else:
                raise ValueError(f"Not implemented classification loss: {self.classification_loss}")

        if 'prediction' in self.task:
            if self.prediction_loss == 'MAE':
                self.prediction_loss_fn = nn.L1Loss()
            elif self.prediction_loss == 'MSE':
                self.prediction_loss_fn = nn.MSELoss()
            elif self.prediction_loss == 'CE':
                self.prediction_loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Not implemented prediction loss: {self.prediction_loss}")

        if hasattr(self, 'detection_loss_fn'):
            print(f"Use loss {self.detection_loss_fn} for detection")
        elif hasattr(self, 'onset_detection_loss_fn'):
            print(f"Use loss {self.onset_detection_loss_fn} for onset detection")
        elif hasattr(self, 'classification_loss_fn'):
            print(f"Use loss {self.classification_loss_fn} for classification")

        if hasattr(self, 'prediction_loss_fn'):
            print(f"Use loss {self.prediction_loss_fn} for prediction")

    def forward(self, z, label, truth):
        """
        :param z: tuple(cls_prob, pred_value)
        :param label: cls_truth
        :param truth: pred_truth
        """
        p, y = z

        loss = 0.
        if 'detection' in self.task:
            loss += self.detection_loss_fn(input=p, target=label)
        elif 'classification' in self.task:
            loss += self.classification_loss_fn(input=p, target=label.long())
        elif 'onset_detection' in self.task:
            loss += self.onset_detection_loss_fn(input=p, target=label)

        if 'prediction' in self.task:
            y = self.scaler.inv_transform(y)
            truth = self.scaler.inv_transform(truth)
            loss += self.lamb * self.prediction_loss_fn(input=y, target=truth)

        return loss
