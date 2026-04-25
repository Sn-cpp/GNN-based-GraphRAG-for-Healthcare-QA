import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class R_GCN(nn.Module):
    """
    Implementation of Relational-Graph Convolutional Network using RGCNConv
    """
    
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

        # Relation embedding
        self.rel_emb = nn.Embedding(num_relations, out_dim)

        # Projection from text space into structural space
        self.query_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        """
        Function to perform forward pass
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_type)
            if i != len(self.convs) - 1:
                h = F.relu(h)

        # Normalize for cosine space
        h = F.normalize(h, dim=1)
        return h

    # Build query embedding: (source node + relation)
    def build_query(self, h, src, rel):
        return F.normalize(h[src] + self.rel_emb(rel), dim=1)

    # Text to structural query
    def project_query(self, q_text):
        return F.normalize(self.query_proj(q_text), dim=1)

    # Cosine scoring, align the GNN with retrieval process
    def score_cosine(self, q, h_nodes):
        return F.cosine_similarity(q, h_nodes, dim=-1)