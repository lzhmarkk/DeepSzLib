import torch
import torch.nn as nn
from models.DCRNN.DCRNNEncoder import DCRNNEncoder
from models.DCRNN.DCRNNDecoder import DCRNNDecoder
from models.utils import Segmentation
from models.DCRNN.graph import distance_support, correlation_support, norm_graph


class DCRNN(nn.Module):
    def __init__(self, args):
        super(DCRNN, self).__init__()

        self.num_nodes = args.n_channels
        self.num_rnn_layers = args.layers
        self.rnn_units = args.hidden
        self.preprocess = args.preprocess
        self.seg = args.seg
        self.dcgru_activation = args.dcgru_activation
        self.filter_type = args.filter_type
        self.use_support = args.use_support
        self.horizon = args.horizon // args.seg

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.hidden, self.num_nodes)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2

        self.encoder = DCRNNEncoder(input_dim=self.dim,
                                    max_diffusion_step=args.max_diffusion_step,
                                    hid_dim=self.rnn_units, num_nodes=self.num_nodes,
                                    num_rnn_layers=self.num_rnn_layers,
                                    dcgru_activation=self.dcgru_activation,
                                    filter_type=self.filter_type)

        self.fc = nn.Linear(self.rnn_units, 1)
        self.dropout = nn.Dropout(args.dropout)

        self.task = args.task
        assert 'cls' in self.task or 'anomaly' in self.task

        if 'pred' in self.task:
            self.decoder = DCRNNDecoder(input_dim=self.dim,
                                        max_diffusion_step=args.max_diffusion_step,
                                        hid_dim=self.rnn_units, num_nodes=self.num_nodes,
                                        output_dim=self.dim,
                                        num_rnn_layers=self.num_rnn_layers,
                                        dcgru_activation=self.dcgru_activation,
                                        filter_type=self.filter_type)

    def get_support(self, x):
        if self.use_support == 'dist':
            if not hasattr(self, 'supports'):
                support = distance_support(self.num_nodes)
                support = norm_graph(support, self.filter_type)
                support = [s.to(x.device) for s in support]
                self.supports = support
            supports = self.supports
        elif self.use_support == 'corr':
            supports = []
            for _x in x:
                support = correlation_support(_x.cpu().numpy())
                support = norm_graph(support, self.filter_type)
                support = torch.stack([s.to(x.device) for s in support], dim=0)
                supports.append(support)
            supports = torch.stack(supports, dim=0).transpose(1, 0)

        else:
            raise ValueError()

        return supports

    def forward(self, x, p, y):
        # (B, T, C, D)
        input_seq = x
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        supports = self.get_support(x)

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).to(x.device)

        # last hidden state of the encoder is the context
        # (num_rnn_layers, batch, rnn_units*num_nodes), (max_seq_len, batch, rnn_units*num_nodes)
        final_hidden, output = self.encoder(input_seq, init_hidden_state, supports)

        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(output, dim0=0, dim1=1)

        if 'cls' in self.task:
            last_out = output[:, -1, :]  # extract last relevant output
            last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)  # (batch_size, num_nodes, rnn_units)
            logits = self.fc(torch.relu(self.dropout(last_out))).squeeze(dim=-1)  # final FC layer, (B, C)
            pool_logits, _ = torch.max(logits, dim=1)  # max-pooling over nodes, (batch_size, num_classes)

        else:
            last_out = output.view(batch_size, max_seq_len, self.num_nodes, self.rnn_units)  # (batch_size, num_nodes, rnn_units)
            logits = self.fc(torch.relu(self.dropout(last_out))).squeeze(dim=-1)  # final FC layer, (B, T, C)
            pool_logits, _ = torch.max(logits, dim=2)  # max-pooling over nodes, (batch_size, T, num_classes)

        if 'pred' not in self.task:
            return pool_logits, _
        else:
            # (seq_len, batch_size, num_nodes * output_dim)
            outputs = self.decoder(y.transpose(0, 1), final_hidden, supports, teacher_forcing_ratio=None)
            # (seq_len, batch_size, num_nodes, output_dim)
            outputs = outputs.reshape(self.horizon, batch_size, self.num_nodes, self.dim)
            # (batch_size, seq_len, num_nodes, output_dim)
            outputs = outputs.transpose(0, 1)

            return pool_logits, outputs
