import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .esg_utils import Dilated_Inception, MixProp, LayerNorm
from .graph import EvolvingGraphLearner
from .static_feat import NodeFeaExtractor
from models.utils import Segmentation


class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout: float):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.dropout = dropout

    def forward(self, x: Tensor):
        _filter = self.filter_conv(x)
        filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        gate = torch.sigmoid(_gate)
        x = filter * gate
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Evolving_GConv(nn.Module):
    def __init__(self, conv_channels: int, residual_channels: int, gcn_depth: int, st_embedding_dim: int,
                 dy_embedding_dim: int, dy_interval: int, dropout=0.3, propalpha=0.05):
        super(Evolving_GConv, self).__init__()
        self.linear_s2d = nn.Linear(st_embedding_dim, dy_embedding_dim)
        self.scale_spc_EGL = EvolvingGraphLearner(conv_channels, dy_embedding_dim)
        self.dy_interval = dy_interval

        self.gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)

    def forward(self, x, st_node_fea):
        b, _, n, t = x.shape
        dy_node_fea = self.linear_s2d(st_node_fea).unsqueeze(0)
        states_dy = dy_node_fea.repeat(b, 1, 1)  # [B, N, C]

        x_out = []

        for i_t in range(0, t, self.dy_interval):
            x_i = x[..., i_t:min(i_t + self.dy_interval, t)]

            input_state_i = torch.mean(x_i.transpose(1, 2), dim=-1)

            dy_graph, states_dy = self.scale_spc_EGL(input_state_i, states_dy)
            x_out.append(self.gconv(x_i, dy_graph))

        x_out = torch.cat(x_out, dim=-1)  # [B, c_out, N, T]
        return x_out


class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, gcn_depth: int,
                 st_embedding_dim, dy_embedding_dim,
                 skip_channels: int, t_len: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout: float, dy_interval: int):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))

        self.s_conv = Evolving_GConv(conv_channels, residual_channels, gcn_depth, st_embedding_dim, dy_embedding_dim,
                                     dy_interval, dropout, propalpha)

        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))

        self.norm = LayerNorm((residual_channels, num_nodes, t_len), elementwise_affine=layer_norm_affline)

    def forward(self, x: Tensor, st_node_fea: Tensor):
        residual = x  # [B, F, N, T]
        # dilated convolution
        x = self.t_conv(x)
        # parametrized skip connection
        skip = self.skip_conv(x)
        # graph convolution
        x = self.s_conv(x, st_node_fea)
        # residual connection
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        return x, skip


class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len: int, kernel_set, dilation_exp: int, n_layers: int, residual_channels: int, conv_channels: int,
                 gcn_depth: int, st_embedding_dim, dy_embedding_dim, skip_channels: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout: float, dy_interval: int):
        super(Block, self).__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1 + block_id * (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1))
        else:
            rf_block = block_id * n_layers * (kernel_size - 1) + 1

        dilation_factor = 1
        for i in range(1, n_layers + 1):
            if dilation_exp > 1:
                rf_size_i = int(rf_block + (kernel_size - 1) * (dilation_exp ** i - 1) / (dilation_exp - 1))
            else:
                rf_size_i = rf_block + i * (kernel_size - 1)
            t_len_i = total_t_len - rf_size_i + 1

            self.append(
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, st_embedding_dim, dy_embedding_dim,
                          skip_channels, t_len_i, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval[i - 1])
            )
            dilation_factor *= dilation_exp

    def forward(self, x: Tensor, st_node_fea: Tensor, skip_list):
        flag = 0
        for layer in self:
            flag += 1
            x, skip = layer(x, st_node_fea)
            skip_list.append(skip)
        return x, skip_list


class ESG(nn.Module):
    def __init__(self, args):
        super(ESG, self).__init__()
        self.n_blocks = 1
        self.dropout = args.dropout
        self.num_nodes = args.n_channels
        self.st_embedding_dim = args.st_embedding_dim
        self.seq_length = args.window // args.seg
        self.hidden = args.hidden
        self.seg = args.seg
        self.preprocess = args.preprocess
        self.n_layers = args.layers
        self.kernel_set = args.kernel_set
        self.gcn_depth = args.gcn_depth
        self.dy_embedding_dim = args.dy_embedding_dim
        self.propalpha = args.propalpha
        self.dy_interval = args.dy_interval

        if self.preprocess == 'seg':
            self.dim = self.hidden
            self.segmentation = Segmentation(self.seg, self.dim, self.num_nodes)
        elif self.preprocess == 'fft':
            self.dim = self.seg // 2
            self.fc = nn.Linear(self.dim, self.hidden)

        kernel_size = self.kernel_set[-1]
        self.receptive_field = self.n_layers * (kernel_size - 1) + 1
        self.total_t_len = max(self.receptive_field, self.seq_length)

        self.start_conv = nn.Conv2d(self.dim, self.hidden, kernel_size=(1, 1))
        self.blocks = Block(0, self.total_t_len, self.kernel_set, 1, self.n_layers, self.hidden, self.hidden, self.gcn_depth,
                            self.st_embedding_dim, self.dy_embedding_dim, 2 * self.hidden, self.num_nodes, True, self.propalpha, self.dropout, self.dy_interval)

        self.skip0 = nn.Conv2d(self.dim, 2 * self.hidden, kernel_size=(1, self.total_t_len), bias=True)
        self.skipE = nn.Conv2d(self.hidden, 2 * self.hidden, kernel_size=(1, self.total_t_len - self.receptive_field + 1), bias=True)

        self.decoder = nn.Sequential(nn.Linear(2 * self.num_nodes * self.hidden, self.hidden),
                                     nn.GELU(),
                                     nn.Linear(self.hidden, 1))

        self.stfea_encode = NodeFeaExtractor(self.st_embedding_dim, self.num_nodes)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        if self.preprocess == 'seg':
            x = self.segmentation.segment(x)  # (B, T, C, D)
        elif self.preprocess == 'fft':
            pass  # (B, T, C, D)

        x = x.transpose(3, 1)  # (bs, input_dim, num_nodes, window)
        if self.seq_length < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))

        st_node_fea = self.stfea_encode()

        skip_list = [self.skip0(F.dropout(x, self.dropout, training=self.training))]
        x = self.start_conv(x)

        x, skip_list = self.blocks(x, st_node_fea, skip_list)

        skip_list.append(self.skipE(x))
        skip_list = torch.cat(skip_list, -1)  # [B, skip_channels, N, n_layers+2]

        z = torch.sum(skip_list, dim=3, keepdim=True)  # [B, skip_channels, N, 1]

        z = F.relu(z)
        z = z.reshape(bs, -1)
        z = self.decoder(z).squeeze(dim=-1)
        return z, None
