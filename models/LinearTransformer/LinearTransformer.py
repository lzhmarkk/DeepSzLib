import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import check_tasks


class RecurrentLinearAttention(nn.Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    Modified from https://github.com/idiap/fast-transformers/
    """

    def __init__(self, dimensions, heads, eps=1e-6):
        super(RecurrentLinearAttention, self).__init__()
        self.eps = eps
        self.dimensions = dimensions
        self.heads = heads

        self.forward_query = nn.Linear(dimensions, dimensions)
        self.forward_key = nn.Linear(dimensions, dimensions)
        self.forward_value = nn.Linear(dimensions, dimensions)

    def forward(self, query, key, value, Si, Zi):
        # Apply the feature map to the query and key
        Q = self.forward_query(query).reshape(*query.shape[:-1], self.heads, self.dimensions // self.heads)
        K = self.forward_key(key).reshape(*key.shape[:-1], self.heads, self.dimensions // self.heads)
        V = self.forward_value(value).reshape(*value.shape[:-1], self.heads, self.dimensions // self.heads)

        # feature mapping
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # Extract some shapes
        N, H, D = Q.shape
        _, _, M = V.shape

        # Extract the memory or initialize it
        if Si is None and Zi is None:
            Si = query.new_zeros((N, H, D, M))
            Zi = query.new_zeros((N, H, D))

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Update the internal state
        #
        # NOTE: The if clause is added due to GitHub PR #10. Simply using the
        # following two lines does not perform the operation in place which
        # means it is slower for inference.
        if K.grad_fn is not None or V.grad_fn is not None:
            Zi = Zi + K
            Si = Si + torch.einsum("nhd,nhm->nhdm", K, V)
        else:
            Zi += K
            Si += torch.einsum("nhd,nhm->nhdm", K, V)

        # Compute the output
        Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)

        V = V.reshape(*query.shape[:-1], self.dimensions)
        return V, Si, Zi


class LinearTransformer(nn.Module):
    supported_tasks = ['detection', 'onset_detection', 'classification']
    unsupported_tasks = ['prediction']

    def __init__(self, args):
        super().__init__()
        self.patch_len = args.patch_len
        self.window = args.window
        self.horizon = args.horizon
        self.hidden = args.hidden
        self.layers = args.layers
        self.channels = args.n_channels
        self.heads = args.heads
        self.dropout = args.dropout
        self.preprocess = args.preprocess
        self.task = args.task
        self.seq_len = self.window // self.patch_len
        self.task = args.task
        check_tasks(self)

        self.dim = self.patch_len // 2
        self.fc = nn.Linear(self.channels * self.dim, self.hidden)

        self.msa = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for _ in range(self.layers):
            self.msa.append(RecurrentLinearAttention(self.hidden, 4))
            self.ffn.append(nn.Sequential(nn.Linear(self.hidden, 4 * self.hidden), nn.GELU(), nn.Linear(4 * self.hidden, self.hidden)))
            self.norm1.append(nn.LayerNorm(self.hidden))
            self.norm2.append(nn.LayerNorm(self.hidden))

        self.decoder = nn.Linear(self.hidden, args.n_classes)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, p, y):
        # (B, T, C, D/S)
        bs = x.shape[0]

        x = self.fc(x.reshape(bs, self.seq_len, self.channels * self.dim))  # (B, T, D)

        h = x
        for layer in range(self.layers):
            # MSA
            Si, Zi = None, None
            output = []
            for t in range(self.seq_len):
                ht = h[:, t, :]
                ht, Si, Zi = self.msa[layer](ht, ht, ht, Si, Zi)
                output.append(ht)
            h = self.norm1[layer](h + self.dropout(torch.stack(output, dim=1)))  # (B, T, D)

            # FFN
            h = self.norm2[layer](h + self.dropout(self.ffn[layer](h)))

        # decoder
        if 'onset_detection' in self.task:
            z = h  # (B, T, D)
            z = self.decoder(z).squeeze(dim=-1)  # (B, T)
        elif 'detection' in self.task or 'classification' in self.task:
            z = h[:, -1, :]  # (B, D)
            z = self.decoder(z).squeeze(dim=-1)  # (B)
        else:
            raise NotImplementedError

        return z, None
