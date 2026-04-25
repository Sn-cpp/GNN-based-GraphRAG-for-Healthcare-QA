import torch
import numpy as np
import pandas as pd


class KnowledgeGraph:
    """
    Construct a knowledge graph from triplets
    """
    
    def __init__(self, triplets: pd.DataFrame, node2id: dict):
        self.df = triplets.copy()

        self.node2id = node2id
        self.rel2id = None
        self.num_relations = 0

        self.edge_index = None
        self.edge_type = None

    # ---------- MAPPINGS ----------
    def build_mappings(self):
        """
        Function to build mappings for relations
        """

        # Relations
        relations = sorted(self.df['Predicate'].unique())
        self.rel2id = {rel: i for i, rel in enumerate(relations)}

    def build_adjacency(self):
        """
        Function to build fast lookup for neighbors
        """
        self.adj = {}

        for s, r, o in self.df.itertuples(index=False):
            self.adj.setdefault(s, []).append((s, r, o))

    # ---------- GRAPH ----------
    def build_graph(self, add_inverse_edges: bool = True):
        """
        Function to build PyG-compatible graph
        """

        if self.rel2id is None:
            self.build_mappings()

        # Map to IDs
        src = self.df['Subject'].map(self.node2id).values
        dst = self.df['Object'].map(self.node2id).values
        rel = self.df['Predicate'].map(self.rel2id).values

        edge_index = np.vstack([src, dst])
        edge_type = rel

        self.num_relations = len(self.rel2id)

        if add_inverse_edges:
            inv_relation_offset = self.num_relations # New IDs range
            inv_edge_index = np.vstack([dst, src])
            inv_edge_type = rel + inv_relation_offset # Assign new IDs

            edge_index = np.hstack([edge_index, inv_edge_index])
            edge_type = np.concatenate([edge_type, inv_edge_type])

            # Update the number of relations
            self.num_relations *= 2

        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_type = torch.tensor(edge_type, dtype=torch.long)

        return self.edge_index, self.edge_type

    # ---------- GETTERS ----------
    def get_neighbors(self, node):
        return self.adj.get(node, [])

    def get_num_nodes(self):
        return len(self.node2id)

    def get_num_relations(self):
        return self.num_relations

    def get_data(self):
        return {
            "node2id": self.node2id,
            "rel2id": self.rel2id,
            "num_relations": self.num_relations,
            "edge_index": self.edge_index,
            "edge_type": self.edge_type
        }