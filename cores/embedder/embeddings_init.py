import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    """
    Utilize sentence-transformers/all-MiniLM-L6-v2 to generate initial embeddings for R-GCN from triplets
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=torch.device('cpu'), max_length: int = 128):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./hf_cache')
        self.model = AutoModel.from_pretrained(model_name, cache_dir='./hf_cache').to(self.device)
        self.model.eval()

        self.max_length = max_length

    def load_triplets(self, path: str):
        """
        Function to read triplets from csv file
        """
        
        return pd.read_csv(
            path,
            names=["Subject", "Predicate", "Object"]
        )

    def build_node_texts(self, df: pd.DataFrame, max_context: int = 50):
        """
        Build textual context per node from triplets (directional only)
        """

        rel_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for s, r, o in df.itertuples(index=False):
            # Forward direction
            rel_dict[s][o]['true'].append(f"{r} {o}")

            # Construct a simple placeholder for the inverse
            rel_dict[o][s]['placeholder'].append(f" related to {s} via '{r}'")


        # Add identity + limit context size
        context = dict()
        for src, dst_dict in rel_dict.items():
            context_list = []
            for dst, rels in dst_dict.items():
                if len(rels['true']) == 0:
                    context_list.extend(rels['placeholder'])
                else:
                    context_list.extend(rels['true'])
            
            context[src] = f"{src} " + ", ".join(context_list[:max_context]) 

        return context
        

    # ---------- ENCODING ----------
    def _mean_pool(self, last_hidden, attention_mask):
        # Compute mean-pooled sentence embedding
        mask = attention_mask.unsqueeze(-1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = 32):
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
                pooled_cpu = pooled.cpu()

                del encoded, outputs, pooled
                
                all_embeddings.append(pooled_cpu)

        torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0)

    # ---------- MAIN ----------
    def run(self, path: str):
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

        embeddings = self.encode(texts).numpy()

        node2id = {node: i for i, node in enumerate(nodes)}
        h_text = torch.tensor(embeddings, dtype=torch.float32)

        return {
            "df": df,
            "node_texts": node_texts,
            "node2id": node2id,
            "h_text": h_text  # node feature matrix
        }
    
    def unload_model(self):
        """
        Safely unload model from GPU and free memory.
        Does NOT affect already computed embeddings.
        """

        if hasattr(self, "model") and self.model is not None:
            try:
                # Move model back to CPU first (safer than direct delete)
                self.model.to("cpu")
            except Exception:
                pass

            # Delete model and tokenizer references
            del self.model
            self.model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()