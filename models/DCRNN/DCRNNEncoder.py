import torch
import torch.nn as nn
from models.DCRNN.DCRNNCell import DCGRUCell


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, hid_dim, num_nodes, num_rnn_layers, dcgru_activation=None, filter_type='laplacian'):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes,
                                        nonlinearity=dcgru_activation,
                                        filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes,
                                            nonlinearity=dcgru_activation,
                                            filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []

        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []

            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](supports, current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)

            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0)  # (seq_len, batch_size, num_nodes * rnn_units)

        output_hidden = torch.stack(output_hidden, dim=0)  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))

        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)
