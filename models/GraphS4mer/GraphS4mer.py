from torch_geometric.nn import (
    GINEConv,
    SAGEConv,
)
import torch_geometric
from .graph_learner import *
from .s4 import S4Model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.DCRNN.graph import distance_support, norm_graph
from models.utils import check_tasks


def calculate_cosine_decay_weight(max_weight, epoch, epoch_total, min_weight=0):
    """
    Calculate decayed weight (hyperparameter) based on cosine annealing schedule
    Referred to https://arxiv.org/abs/1608.03983
    and https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    curr_weight = min_weight + \
                  0.5 * (max_weight - min_weight) * (1 + math.cos(epoch / epoch_total * math.pi))
    return curr_weight


def calculate_normalized_laplacian(adj):
    """
    Args:
        adj: torch tensor, shape (batch, num_nodes, num_nodes)

    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    D = diag(A)
    """

    batch, num_nodes, _ = adj.shape
    d = adj.sum(-1)  # (batch, num_nodes)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (batch, num_nodes, num_nodes)

    identity = (torch.eye(num_nodes).unsqueeze(0).repeat(batch, 1, 1)).to(
        adj.device
    )  # (batch, num_nodes, num_nodes)
    normalized_laplacian = identity - torch.matmul(
        torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt
    )

    return normalized_laplacian


def feature_smoothing(adj, X):
    # normalized laplacian
    L = calculate_normalized_laplacian(adj)

    feature_dim = X.shape[-1]
    mat = torch.matmul(torch.matmul(X.transpose(1, 2), L), X) / (feature_dim ** 2)
    loss = mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
    return loss


def get_knn_graph(x, k, dist_measure="cosine", undirected=True):
    if dist_measure == "euclidean":
        dist = torch.cdist(x, x, p=2.0)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        knn_val, knn_ind = torch.topk(
            dist, k, dim=-1, largest=False
        )  # smallest distances
    elif dist_measure == "cosine":
        norm = torch.norm(x, dim=-1, p="fro")[:, :, None]
        x_norm = x / norm
        dist = torch.matmul(x_norm, x_norm.transpose(1, 2))
        knn_val, knn_ind = torch.topk(
            dist, k, dim=-1, largest=True
        )  # largest similarities
    else:
        raise NotImplementedError

    adj_mat = (torch.ones_like(dist) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)

    adj_mat = torch.clamp(adj_mat, min=0.0)  # remove negatives

    if undirected:
        adj_mat = (adj_mat + adj_mat.transpose(1, 2)) / 2

    # add self-loop
    I = (
        torch.eye(adj_mat.shape[-1], adj_mat.shape[-1])
        .unsqueeze(0)
        .repeat(adj_mat.shape[0], 1, 1)
        .to(bool)
    ).to(x.device)
    adj_mat = adj_mat * (~I) + I

    # to sparse graph
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

    return edge_index, edge_weight, adj_mat


def prune_adj_mat(adj_mat, num_nodes, method="thresh", edge_top_perc=None, knn=None, thresh=None):
    if method == "thresh":
        sorted, indices = torch.sort(
            adj_mat.reshape(-1, num_nodes * num_nodes),
            dim=-1,
            descending=True,
        )
        K = int((num_nodes ** 2) * edge_top_perc)
        mask = adj_mat > sorted[:, K].unsqueeze(1).unsqueeze(2)
        adj_mat = adj_mat * mask
    elif method == "knn":
        knn_val, knn_ind = torch.topk(
            adj_mat, knn, dim=-1, largest=True
        )
        adj_mat = (torch.ones_like(adj_mat) * 0).scatter_(-1, knn_ind, knn_val).to(adj_mat.device)
    elif method == "thresh_abs":
        mask = (adj_mat > thresh).float()
        adj_mat = adj_mat * mask
    else:
        raise NotImplementedError

    return adj_mat


class GraphS4mer(nn.Module):
    supported_tasks = ['detection', 'classification']
    unsupported_tasks = ['prediction', 'onset_detection']

    def __init__(self, args):
        super().__init__()

        self.input_dim = 1
        self.num_nodes = args.n_channels
        self.channels = 1
        self.dropout = args.dropout
        self.edge_top_perc = args.edge_top_perc
        self.graph_pool = args.graph_pool
        self.hidden_dim = args.hidden
        self.state_dim = args.hidden // 2
        self.K = args.K
        self.regularizations = args.regularizations
        self.metric = args.graph_metric
        self.undirected_graph = args.undirected_graph
        self.residual_weight = args.residual_weight
        self.use_prior = args.use_prior
        self.temporal_pool = args.temporal_pool
        self.temporal_model = args.temporal_model
        self.max_seq_len = args.window
        self.resolution = args.seg * 10
        self.prune_method = args.prune_method
        self.thresh = args.thresh
        self.g_conv = args.g_conv
        self.num_gnn_layers = args.num_gnn_layers
        self.num_temporal_layers = args.num_temporal_layers
        self.decay_residual_weight = args.decay_residual_weight
        self.prenorm = args.prenorm
        self.bidirectional = args.bidirectional
        self.postact = args.postact
        self.adj_embed_dim = args.adj_embed_dim
        self.gin_mlp = args.gin_mlp
        self.train_eps = args.train_eps
        self.activation_fn = args.activation_fn
        assert args.preprocess == 'raw'
        self.task = args.task
        check_tasks(self)

        # temporal layer
        if self.temporal_model == "gru":
            self.t_model = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_temporal_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        elif self.temporal_model == "s4":
            self.t_model = S4Model(
                d_input=self.input_dim,
                d_model=self.hidden_dim,
                d_state=self.state_dim,
                channels=self.channels,
                n_layers=self.num_temporal_layers,
                dropout=self.dropout,
                prenorm=self.prenorm,
                l_max=self.max_seq_len,
                bidirectional=self.bidirectional,
                postact=self.postact,  # none or 'glu'
                add_decoder=False,
                pool=False,  # hard-coded
                temporal_pool=None,
            )

        else:
            raise NotImplementedError

        # graph learning layer
        self.attn_layers = GraphLearner(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_nodes=self.num_nodes,
            embed_dim=self.adj_embed_dim,
            metric_type=self.metric,
        )

        # gnn layers
        self.gnn_layers = nn.ModuleList()
        if self.g_conv == "graphsage":
            for _ in range(self.num_gnn_layers):
                self.gnn_layers.append(SAGEConv(self.hidden_dim, self.hidden_dim))
        elif self.g_conv == "gine":
            for _ in range(self.num_gnn_layers):
                if self.gin_mlp:
                    gin_nn = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                else:
                    gin_nn = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.gnn_layers.append(
                    GINEConv(
                        nn=gin_nn, eps=0.0, train_eps=self.train_eps, edge_dim=1
                    )
                )
        else:
            raise NotImplementedError

        if self.activation_fn == "relu":
            self.activation = nn.ReLU()
        elif self.activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif self.activation_fn == "elu":
            self.activation = nn.ELU()
        elif self.activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=self.dropout)
        self.classifier = nn.Linear(self.hidden_dim, args.n_classes)

    def forward(self, x, p, y):
        # (B, T, C, S)
        lengths = None
        bs = x.shape[0]
        x = x.transpose(2, 1).reshape(bs * self.num_nodes, self.max_seq_len, 1)  # (B*C, T, 1)

        x = x.transpose(2, 1).reshape(bs * self.num_nodes, self.max_seq_len, 1)  # (B*C, T, 1)
        batch = x.shape[0] // self.num_nodes
        num_nodes = self.num_nodes
        _, seq_len, _ = x.shape

        if lengths is not None:
            lengths = torch.repeat_interleave(lengths, num_nodes, dim=0)

        # temporal layer
        if self.temporal_model == "s4":
            x = self.t_model(x, lengths)  # (batch * num_nodes, seq_len, hidden_dim)
        else:
            if lengths is not None:
                x = pack_padded_sequence(x, lengths, batch_first=True)
            x, _ = self.t_model(x)
            if lengths is not None:
                x, lengths = pad_packed_sequence(x)

        # get output with <resolution> as interval
        if lengths is None:
            x = x.view(
                batch, num_nodes, seq_len, -1
            )  # (batch, num_nodes, seq_len, hidden_dim)
            x_tmp = []
            num_dynamic_graphs = self.max_seq_len // self.resolution
            for t in range(num_dynamic_graphs):
                start = t * self.resolution
                stop = start + self.resolution
                curr_x = torch.mean(x[:, :, start:stop, :], dim=2)
                x_tmp.append(curr_x)
            x_tmp = torch.stack(
                x_tmp, dim=1
            )  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)
            x = x_tmp.reshape(
                -1, num_nodes, self.hidden_dim
            )  # (batch * num_dynamic_graphs, num_nodes, hidden_dim)
            del x_tmp
        else:  # for variable lengths, mean pool over actual lengths
            x = torch.stack(
                [
                    torch.mean(out[:length, :], dim=0)
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
            x = x.reshape(batch, num_nodes, -1)  # (batch, num_nodes, hidden_dim)
            num_dynamic_graphs = 1

        # get initial adj
        if self.use_prior:
            # adj_mat = torch_geometric.utils.to_dense_adj(
            #     edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr
            # )
            adj_mat = distance_support(self.num_nodes)
            adj_mat = norm_graph(adj_mat, 'laplacian')
        else:
            # knn cosine graph
            edge_index, edge_weight, adj_mat = get_knn_graph(
                x,
                self.K,
                dist_measure="cosine",
                undirected=self.undirected_graph,
            )
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)
            adj_mat = adj_mat.to(x.device)

        # learn adj mat
        attn_weight = self.attn_layers(
            x
        )  # (batch*num_dynamic_graphs, num_nodes, num_nodes)

        # to undirected
        if self.undirected_graph:
            attn_weight = (attn_weight + attn_weight.transpose(1, 2)) / 2
        raw_attn_weight = attn_weight.clone()

        # add residual
        if len(adj_mat.shape) == 2:
            adj_mat = torch.cat([adj_mat] * num_dynamic_graphs * batch, dim=0)
        elif len(adj_mat.shape) == 3 and (adj_mat.shape != attn_weight.shape):
            adj_mat = torch.cat([adj_mat] * num_dynamic_graphs, dim=0)

        # knn graph weight (aka residual weight) decay
        if self.decay_residual_weight:
            epoch, epoch_total = None, None
            assert (epoch is not None) and (epoch_total is not None)
            residual_weight = calculate_cosine_decay_weight(
                max_weight=self.residual_weight, epoch=epoch, epoch_total=epoch_total, min_weight=0
            )
        else:
            residual_weight = self.residual_weight
        # add knn graph
        adj_mat = (
                residual_weight * adj_mat + (1 - residual_weight) * attn_weight
        )

        # prune graph
        adj_mat = prune_adj_mat(
            adj_mat,
            num_nodes,
            method=self.prune_method,
            edge_top_perc=self.edge_top_perc,
            knn=self.K,
            thresh=self.thresh,
        )

        # regularization loss
        reg_losses = self.regularization_loss(x, adj=adj_mat)

        spatial_x = []
        for t in range(batch * num_dynamic_graphs):
            # back to sparse graph
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat[t])
            _x = x[t]

            # add self-loop
            edge_index, edge_weight = torch_geometric.utils.remove_self_loops(
                edge_index=edge_index, edge_attr=edge_weight
            )
            edge_index, edge_weight = torch_geometric.utils.add_self_loops(
                edge_index=edge_index,
                edge_attr=edge_weight,
                fill_value=1,
            )

            _x = _x.view(
                num_nodes, -1
            )  # (num_nodes, hidden_dim)

            for i in range(len(self.gnn_layers)):
                # gnn layer
                _x = self.gnn_layers[i](
                    _x, edge_index=edge_index, edge_attr=edge_weight.reshape(-1, 1)
                )
                _x = self.dropout(
                    self.activation(_x)
                )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
            _x = _x.view(num_nodes, -1)  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)
            spatial_x.append(_x)
        x = torch.stack(spatial_x, dim=0).reshape(batch, num_dynamic_graphs, num_nodes, -1)

        # temporal pool
        if self.temporal_pool == "last":
            x = x[:, -1, :, :]  # (batch, num_nodes, hidden_dim)
        elif self.temporal_pool == "mean":
            x = torch.mean(x, dim=1)
        else:
            raise NotImplementedError

        # graph pool
        if self.graph_pool == "sum":
            x = torch.sum(x, dim=1)  # (batch, hidden_dim)
        elif self.graph_pool == "mean":
            x = torch.mean(x, dim=1)
        elif self.graph_pool == "max":
            x, _ = torch.max(x, dim=1)
        else:
            raise NotImplementedError
        feat = x.clone()

        # classifier
        x = self.classifier(x).squeeze(dim=-1)

        return x, None

    def regularization_loss(self, x, adj, reduce="mean"):
        """
        Referred to https://github.com/hugochan/IDGL/blob/master/src/core/model_handler.py#L1116
        """
        batch, num_nodes, _ = x.shape
        n = num_nodes

        loss = {}

        if "feature_smoothing" in self.regularizations:
            curr_loss = feature_smoothing(adj=adj, X=x) / (n ** 2)
            if reduce == "mean":
                loss["feature_smoothing"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["feature_smoothing"] = torch.sum(curr_loss)
            else:
                loss["feature_smoothing"] = curr_loss

        if "degree" in self.regularizations:
            ones = torch.ones(batch, num_nodes, 1).to(x.device)
            curr_loss = -(1 / n) * torch.matmul(
                ones.transpose(1, 2), torch.log(torch.matmul(adj, ones))
            ).squeeze(-1).squeeze(-1)
            if reduce == "mean":
                loss["degree"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["degree"] = torch.sum(curr_loss)
            else:
                loss["degree"] = curr_loss

        if "sparse" in self.regularizations:
            curr_loss = (
                    1 / (n ** 2) * torch.pow(torch.norm(adj, p="fro", dim=(-1, -2)), 2)
            )

            if reduce == "mean":
                loss["sparse"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["sparse"] = torch.sum(curr_loss)
            else:
                loss["sparse"] = curr_loss

        if "symmetric" in self.regularizations and self.undirected_graph:
            curr_loss = torch.norm(adj - adj.transpose(1, 2), p="fro", dim=(-1, -2))
            if reduce == "mean":
                loss["symmetric"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["symmetric"] = torch.sum(curr_loss)
            else:
                loss["symmetric"] = curr_loss

        return loss
