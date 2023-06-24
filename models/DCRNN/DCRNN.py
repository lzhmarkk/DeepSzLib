import torch
import torch.nn as nn
from models.DCRNN.DCRNNEncoder import DCRNNEncoder
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
        self.relu = nn.ReLU()

        self.ff = nn.Parameter(torch.randn([self.num_nodes - 10]))

    def get_support(self, x):
        if self.use_support == 'dist':
            if not hasattr(self, 'supports'):
                support = distance_support(self.num_nodes)
                support = norm_graph(support, self.filter_type)
                support = [s.to(x.device) for s in support]
                self.supports = support
            supports = self.supports
            raise ValueError()
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

    def forward(self, x):
        # (B, T, C, D)
        input_seq = x
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        supports = self.get_support(x)

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).to(x.device)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        _, final_hidden = self.encoder(input_seq, init_hidden_state, supports)
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # extract last relevant output
        last_out = output[:, -1, :]
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out))).squeeze()  # (B, C)

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
