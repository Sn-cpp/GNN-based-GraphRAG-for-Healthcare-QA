import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class R_GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=2):
        super().__init__()

        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        # Output layer
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations))

    def forward(self, x, edge_index, edge_type):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_type)
            if i != len(self.convs) - 1:
                h = F.relu(h)
        return h

    # Dot-product scoring
    def score(self, h, src, dst):
        return (h[src] * h[dst]).sum(dim=1)
    
