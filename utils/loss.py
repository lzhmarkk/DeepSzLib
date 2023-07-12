import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self, cls_loss, pred_loss, multi_task, lamb, scaler):
        super().__init__()
        self.multi_task = multi_task
        self.lamb = lamb
        self.scaler = scaler

        if cls_loss == 'BCE':
            self.cls_loss = nn.BCEWithLogitsLoss()
        elif cls_loss == "WBCE":
            self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9]))
        else:
            raise ValueError(f"Not implemented classification loss: {cls_loss}")

        if self.multi_task:
            if pred_loss == 'MAE':
                self.pred_loss = nn.L1Loss()
            elif pred_loss == 'MSE':
                self.pred_loss = nn.MSELoss()
            elif pred_loss == 'CE':
                self.pred_loss = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Not implemented prediction loss: {pred_loss}")

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

        loss = self.cls_loss(input=p, target=label)

        if self.multi_task:
            y = self.scaler.inv_transform(y)
            truth = self.scaler.inv_transform(truth)
            loss += self.lamb * self.pred_loss(input=y, target=truth)

        return loss


def get_loss(args):
    return MyLoss(args.cls_loss, args.pred_loss, args.multi_task, args.lamb, args.scaler)
