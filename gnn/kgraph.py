import torch
import numpy as np
import pandas as pd


class KnowledgeGraph:
    def __init__(self, triplets: pd.DataFrame, node2id: dict):
        self.df = triplets.copy()

        self.node2id = node2id
        self.rel2id = None

        self.edge_index = None
        self.edge_type = None

    # ---------- MAPPINGS ----------
    def build_mappings(self):
        """
        Build mappings for relations
        """

        # Relations
        relations = sorted(self.df['Predicate'].unique())
        self.rel2id = {rel: i for i, rel in enumerate(relations)}

    def build_adjacency(self):
        """
        Build fast lookup for neighbors
        """
        self.adj = {}

        for s, r, o in self.df.itertuples(index=False):
            self.adj.setdefault(s, []).append((s, r, o))
            self.adj.setdefault(o, []).append((s, r, o))

    # ---------- GRAPH ----------
    def build_graph(self, add_inverse_edges: bool = True):
        """
        Build PyG-compatible graph
        """

        if self.rel2id is None:
            self.build_mappings()

        # Map to IDs
        src = self.df['Subject'].map(self.node2id).values
        dst = self.df['Object'].map(self.node2id).values
        rel = self.df['Predicate'].map(self.rel2id).values

        edge_index = np.vstack([src, dst])
        edge_type = rel

        if add_inverse_edges:
            inv_edge_index = np.vstack([dst, src])
            inv_edge_type = rel  # same relation 

            edge_index = np.hstack([edge_index, inv_edge_index])
            edge_type = np.concatenate([edge_type, inv_edge_type])

        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_type = torch.tensor(edge_type, dtype=torch.long)

        return self.edge_index, self.edge_type

    # ---------- GETTERS ----------
    def get_neighbors(self, node):
        return self.adj.get(node, [])

    def get_num_nodes(self):
        return len(self.node2id)

    def get_num_relations(self):
        return len(self.rel2id)

    def get_data(self):
        return {
            "node2id": self.node2id,
            "rel2id": self.rel2id,
            "edge_index": self.edge_index,
            "edge_type": self.edge_type
        }