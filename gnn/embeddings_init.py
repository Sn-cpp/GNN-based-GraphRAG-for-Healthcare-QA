import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel


class EmbeddingInitializer:
    """
    Utilize sentence-transformers/all-MiniLM-L6-v2 to generate initial embeddings for R-GCN from triplets
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = 'cpu', max_length: int = 128):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./hf_cache')
        self.model = AutoModel.from_pretrained(model_name, cache_dir='./hf_cache').to(self.device)
        self.model.eval()

        self.max_length = max_length

    
    def load_triplets(self, path: str) -> pd.DataFrame:
        """
        Helper function to read triplets from csv file
        """
        
        return pd.read_csv(
            path,
            names=["Subject", "Predicate", "Object"]
        )


    def build_node_texts(self, df: pd.DataFrame, max_context: int = 50) -> dict[str, str]:
        """
        Build textual context per node from triplets (directional only)
        """

        node_texts = defaultdict(list)

        for s, r, o in df.itertuples(index=False):
            s_clean = s.replace("_", " ")
            r_clean = r.replace("_", " ")
            o_clean = o.replace("_", " ")

            # Only forward direction
            node_texts[s].append(f"{s_clean} {r_clean} {o_clean}")

        # Add identity + limit context size
        return {
            node: f"{node.replace('_', ' ')}. " + ". ".join(contexts[:max_context])
            for node, contexts in node_texts.items()
        }

    # ---------- ENCODING ----------
    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embeddings
        """

        all_embeddings = []

        with torch.no_grad():
            # Process by batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = self.model(**encoded)

                pooled = self._mean_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"]
                )

                # Normalize for cosine similarity
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                all_embeddings.append(pooled.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    # ---------- MAIN ----------
    def run(self, path: str) -> dict[str, ]:
        """
        Main pipeline for reading triplets and make embeddings\n
        Return list of triplets, nodes textual context, nodes mapping and nodes embeddings
        """
        
        df = self.load_triplets(path)

        # Build full node set from graph
        all_nodes = pd.concat([df['Subject'], df['Object']]).unique()

        node_texts = self.build_node_texts(df)

        # Ensure ALL nodes exist
        for node in all_nodes:
            node_texts.setdefault(node, node.replace("_", " "))

        nodes = list(all_nodes)
        texts = [node_texts[node] for node in nodes]

        embeddings = self.encode(texts)

        node2id = {node: i for i, node in enumerate(nodes)}
        x = torch.tensor(embeddings, dtype=torch.float32)

        return {
            "df": df,
            "node_texts": node_texts,
            "node2id": node2id,
            "x": x  # node feature matrix
        }