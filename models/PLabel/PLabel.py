import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PLabel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.d_model = args.hidden
        self.seq_len = args.window // args.seg
        self.seg = args.seg
        self.teacher_rate = args.teacher_rate
        self.preprocess = args.preprocess
        self.input_dim = args.input_dim
        self.channels = args.n_channels
        self.method = args.method
        self.temporal_pooling = args.temporal_pooling

        assert self.preprocess == 'fft'
        self.fc = nn.Linear(self.input_dim, self.d_model)

        self.rnn = nn.GRUCell(self.d_model, self.d_model)

        # predict the init label
        self.initial_label_fc = nn.Sequential(nn.Linear(self.channels * self.d_model, self.d_model),
                                              nn.GELU(),
                                              nn.Linear(self.d_model, 1))

        if self.method == 'max':
            self.pred_fc = nn.Linear(self.d_model, 1)
        elif self.method == 'fc':
            self.pred_mlp = nn.Sequential(nn.Linear(self.channels * self.d_model, self.d_model),
                                          nn.GELU(),
                                          nn.Linear(self.d_model, 1))

        self.w = nn.Parameter(torch.randn(self.channels, 1))
        self.w2 = nn.Parameter(torch.randn(1))

        self.label_embedding = nn.Embedding(2, self.d_model)

        # set loss
        args.cls_loss = "BCENoSigmoid"

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        x = self.fc(x)  # (B, T, C, D)

        hidden = torch.zeros(bs * self.channels, self.d_model).to(x.device)
        last_label = None
        all_label = []

        for t in range(self.seq_len):
            _x = x[:, t, :, :].reshape(bs * self.channels, self.d_model)
            hidden = self.rnn(_x, hidden)

            # initial label
            if t == 0:
                last_label = self.initial_label_fc(hidden.reshape(bs, self.channels * self.d_model))
                last_label = torch.sigmoid(last_label).squeeze(-1)

            # curriculum learning
            if self.training and np.random.random() < self.teacher_rate:
                last_label = p

            # calculate current label
            assert (0 <= last_label).all() and (last_label <= 1).all()
            assert last_label.ndim == 1
            label_hidden = (1 - last_label.unsqueeze(1)) * self.label_embedding.weight[0] + last_label.unsqueeze(1) * self.label_embedding.weight[1]
            label_hidden = label_hidden.repeat(self.channels, 1)
            w = self.w.sigmoid().repeat(bs, 1)
            label_hidden = (1 - w) * hidden + w * label_hidden  # (B*C, D)
            # label_hidden = hidden

            if self.method == 'attn':
                label_hidden = label_hidden.reshape(bs, self.channels, self.d_model)
                current_label = torch.softmax(torch.matmul(label_hidden, self.label_embedding.weight.T), dim=-1)[..., 1]
                current_label = torch.max(current_label, dim=1)[0]  # (B)
            elif self.method == 'max':
                label_hidden = label_hidden.reshape(bs, self.channels, self.d_model)
                current_label = self.pred_fc(label_hidden).squeeze(-1)
                current_label = torch.max(current_label, dim=1)[0]  # (B)
                current_label = torch.sigmoid(current_label)
            elif self.method == 'fc':
                label_hidden = label_hidden.reshape(bs, self.channels * self.d_model)
                current_label = self.pred_mlp(label_hidden).squeeze(dim=1)  # (B)
                current_label = torch.sigmoid(current_label)
            else:
                raise ValueError()

            # current_label = (1 - self.w2.sigmoid()) * last_label + self.w2.sigmoid() * current_label
            all_label.append(current_label)

            last_label = current_label

        # temporal pooling
        all_label = torch.stack(all_label, dim=1)  # (B, T)
        assert (0 <= all_label).all() and (all_label <= 1).all()
        assert all_label.ndim == 2

        if self.temporal_pooling == 'max':
            all_label = torch.max(all_label, dim=1)[0]
        elif self.temporal_pooling == 'mean':
            all_label = torch.mean(all_label, dim=1)
        elif self.temporal_pooling == 'last':
            all_label = all_label[:, -1]
        else:
            raise ValueError()

        return all_label, None
