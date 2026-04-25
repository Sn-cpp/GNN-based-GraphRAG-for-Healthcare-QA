import torch
import pandas as pd
import torch.nn.functional as F
from cores.embedder.embeddings_init import TextEmbedder

from cores.kgraph.knowledge_graph import KnowledgeGraph
from online.qwen_model import QwenLLM


class GraphRAGApp:
    """
    Construct a LLM Q&A Application using knowledge graph-assisted Qwen3.5-4B 
    """
    
    def __init__(self, emb_path, csv_path, model_name="sentence-transformers/all-MiniLM-L6-v2",
        topk = 10,
        semantic_contribute=0.5,
        score_tolerance_factor=0.3,
        verbose=False
    ):
        
        # Load embeddings and perform normalization
        data = torch.load(emb_path, map_location=torch.device('cpu'))
        self.x_norm = F.normalize(data["h_text"], dim=1)       # semantic embeddings
        self.h_norm = F.normalize(data["h_struct"], dim=1)    # structural embeddings

        self.node2id = data["node2id"] # nodes mapping

        # Text to Structural projection
        self.query_proj = torch.nn.Linear(384, 384) # query projection layer
        self.query_proj.load_state_dict(data["query_proj"])
        self.query_proj.eval()

        # Reverse mapping
        self.id2node = {v: k for k, v in self.node2id.items()}

        # Load KG
        df = pd.read_csv(csv_path, names=["Subject", "Predicate", "Object"])
        self.kg = KnowledgeGraph(df, self.node2id)
        self.kg.build_graph()
        self.kg.build_adjacency()

        # Load embedding model (to encode the query), embedding dimension is 384 for all-MiniLM-L6-v2 model
        self.query_embedder = TextEmbedder(model_name=model_name)

        self.top_k = topk
        self.sem_contr= semantic_contribute
        self.score_tol_factor = score_tolerance_factor

        # Initialize Qwen
        self.llm = QwenLLM()

        self.verbose = verbose


    # ---------- HYBRID RETRIEVAL ----------
    def retrieve(self, query: str):
        """
        Function to embed the query and retrieve context-relevant nodes\n
        This is equivalent implementation to the BaseRetrieval of LlamaIndex

        Parameters:
            + query: User query sentence
            + top_k: K most relevant nodes
            + alpha: Contribution of the semantic similarity to the hybrid score
        """
        
        # Use the same embedder in training stage to encode the query
        # Encode + normalize query (semantic space)
        q_emb = self.query_embedder.encode([query])[0]
        q_emb = F.normalize(q_emb, dim=0)

        # Project query into structural space
        q_struct = F.normalize(self.query_proj(q_emb), dim=0)

        # Compute similarities
        sim_sem = F.cosine_similarity(q_emb, self.x_norm)
        sim_struct = F.cosine_similarity(q_struct, self.h_norm)

        # Hybrid score
        scores = self.sem_contr * sim_sem + (1 - self.sem_contr) * sim_struct

        topk = torch.topk(scores, k=self.top_k)

        topk_indices = topk.indices.tolist()

        #-----For debugging-----
        if self.verbose:
            # Embedding correlation (global, precomputed ideally)
            diag_corr = (self.x_norm * self.h_norm).sum(dim=1).mean()

            # Score statistics
            sem_mean = sim_sem.mean().item()
            struct_mean = sim_struct.mean().item()

            sem_std = sim_sem.std().item()
            struct_std = sim_struct.std().item()

            # Top-k contribution
            topk_sem = sim_sem[topk.indices]
            topk_struct = sim_struct[topk.indices]

            sem_contrib = topk_sem.mean().item()
            struct_contrib = topk_struct.mean().item()

            # Dominance ratio
            dominance = struct_contrib / (sem_contrib + 1e-8)

            print("=== DIAGNOSTICS ===")
            print(f"Embedding corr (h_struct · h_text): {diag_corr:.4f}")
            print(f"Semantic mean/std: {sem_mean:.4f} / {sem_std:.4f}")
            print(f"Structural mean/std: {struct_mean:.4f} / {struct_std:.4f}")
            print(f"Top-k semantic avg: {sem_contrib:.4f}")
            print(f"Top-k structural avg: {struct_contrib:.4f}")
            print(f"Structural/Semantic ratio: {dominance:.4f}")
            print("===================")

        return topk_indices, scores.detach().cpu()

    # ---------- CONTEXT BUILD ----------
    def build_context(self, nodes: list, scores: torch.Tensor):
        """
        Build KG-based context for LLM with score-aware selection
        """
      
        context = []
        for node_id in nodes:
            node = self.id2node[node_id]
            # Retrieve neighbors of each topK node    
            neighbor_rels = self.kg.get_neighbors(node)


            # Get the corresponding score of each neighbor
            neighbor_scores = torch.tensor([
                scores[self.node2id[o]] for (_, _, o) in neighbor_rels
            ])

            if neighbor_scores.shape[0] == 0:
                continue

            # Compute lower bound scoring threshold to remove irrelevant neighbors in an adaptive way
            # to preserve triplets whose Object's score >= lower_threshold
            mean = neighbor_scores.mean()
            std = neighbor_scores.std()
            lower_threshold = mean + self.score_tol_factor * std

            # Remove triplets with insufficient Object's score
            valid_neighbors = [(r, o) for (s, r, o) in neighbor_rels if scores[self.node2id[o]] >= lower_threshold]

            if len(valid_neighbors) == 0:
                # Fallback: keep at least one fact for this node
                best_idx = torch.argmax(neighbor_scores)
                valid_neighbors = [neighbor_rels[best_idx][1:]]

            # Compress triplets into multi-facts format:
            # + Subject: 
            #       - Predicate_1 Object_1
            #       - Predicate_2 Object_2
            #       - ...
            compressed_fact = "+ " + node.capitalize() + ":\n\t- " + "\n\t- ".join([' '.join(fact) for fact in valid_neighbors])

            context.append(compressed_fact)

        return "\n".join(context)

    # ---------- FULL PIPELINE ----------
    def answer(self, query, stream=True):
        """
        Main function to response a user query
        """
        
        node_ids, scores = self.retrieve(query)
        context = self.build_context(node_ids, scores)

        # -----For debugging-----
        if self.verbose:
            print("===" * 30)
            print("Context:")
            print(context)
            print("===" * 30)

        prompt = f"""
        You are given medical and biological facts:
        {context}

        Question: {query}

        Instructions:
        - Use the facts internally.
        - Use only clear facts.
        - Skip any unclear or inconsistent fact immediately.
        - Do not evaluate, interpret, or reason about facts.
        - Do not mention skipped facts.
        - You may reuse important technical terms from the facts (e.g., domain-specific words).
        - Write a natural answer instead of reproducing the input structure.
        - Do NOT mention the word "fact" or refer to the given data.
        - Answer as if you already know the information.
        - Ignore unclear or inconsistent facts silently.
        - Do not verify or evaluate correctness.
        - If the facts do not cover the question, just express that you haven't taught about it yet.

        Output format (strict): Follow this example exactly
        <answer>
        Children should eat fruits and vegetables. They provide essential nutrients and support healthy growth.
        </answer>

        Rules:
        - No bullet points or lists.
        - No restating input text.
        - No extra text before or after the answer block.

        Start immediately:
        <answer>
        """

        output = self.llm.generate(prompt, stream)

        return output
