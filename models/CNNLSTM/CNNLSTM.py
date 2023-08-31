import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.window // args.seg
        self.channels = args.n_channels
        self.num_classes = 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 23 * ((self.channels - 4) // 2), 512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x, p, y):
        batch, max_seq_len, num_ch, in_dim = x.shape
        x = x.reshape(-1, num_ch, in_dim).unsqueeze(1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)

        out = out.reshape(batch * max_seq_len, -1)
        out = self.fc1(out)
        out = out.reshape(batch, max_seq_len, -1)

        lstm_out, _ = self.lstm(out)
        lstm_out = lstm_out[:, -1, :]
        logits = self.fc2(lstm_out).squeeze(dim=-1)

        return logits, None
