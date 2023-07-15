import torch
import torch.nn as nn


class NodeFeaExtractor(nn.Module):
    def __init__(self, hidden_size_st, num_nodes):
        super(NodeFeaExtractor, self).__init__()

        self.emb = nn.Embedding(num_nodes, hidden_size_st)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size_st)

    def forward(self):
        x = self.emb.weight
        x = torch.relu(x)
        x = self.bn3(x)
        return x
