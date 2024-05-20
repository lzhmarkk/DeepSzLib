import torch
import torch.nn as nn
from models.utils import check_tasks


class DeepSOZ(nn.Module):
    supported_tasks = ['detection', 'onset_detection', 'classification']
    unsupported_tasks = ['prediction']

    def __init__(self, args):
        super().__init__()
        self.window = args.window // args.patch_len
        self.n_channels = args.n_channels
        self.hidden = args.hidden
        self.n_heads = args.n_heads
        self.attn_dropout = args.attn_dropout
        self.rnn_dropout = args.rnn_dropout
        self.preprocess = args.preprocess
        self.patch_len = args.patch_len
        self.onset_history_len = args.onset_history_len
        self.task = args.task
        check_tasks(self)

        assert self.preprocess == 'raw'
        self.pos_encoder = nn.Embedding(self.n_channels + 1, self.patch_len)
        self.tx_encoder = nn.TransformerEncoderLayer(self.patch_len, nhead=self.n_heads, dim_feedforward=256, batch_first=True, dropout=self.attn_dropout)
        self.multi_lstm = nn.LSTM(input_size=self.patch_len, hidden_size=self.hidden,
                                  batch_first=True, bidirectional=True, num_layers=1,
                                  dropout=self.rnn_dropout)
        self.multi_linear = nn.Linear(2 * self.hidden, args.n_classes)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        B = x.shape[0]
        chn_pos = torch.arange(self.n_channels).to(x.device)
        pos_emb = self.pos_encoder(chn_pos)[None, None, :, :]
        h_c = x + pos_emb
        h_m = self.pos_encoder(torch.tensor([self.n_channels] * B * self.window).view(B, self.window, -1).to(x.device))

        # Apply Transformer
        h_c = h_c.reshape(B * self.window, self.n_channels, self.patch_len)
        tx_input = torch.cat((h_c, h_m.reshape(B * self.window, 1, self.patch_len)), dim=1)
        tx_input = self.tx_encoder(tx_input)
        h_c = tx_input[:, :-1, :].view(B, self.window, self.n_channels, self.patch_len)
        h_m = tx_input[:, -1, :].view(B, self.window, -1)

        # Apply multi GRU
        self.multi_lstm.flatten_parameters()
        h_m, _ = self.multi_lstm(h_m)  # (B, T, hidden)

        h_m = self.multi_linear(h_m)  # (B, T, 1)

        # decoder
        if 'onset_detection' in self.task:
            z = h_m.squeeze(dim=-1)  # (B, T)
        elif 'detection' in self.task or 'classification' in self.task:
            z = h_m[:, -1, :].squeeze(dim=-1)
        else:
            raise NotImplementedError

        return z, None
