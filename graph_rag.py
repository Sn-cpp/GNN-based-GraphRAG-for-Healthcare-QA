import torch
import pandas as pd
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModel

from gnn.kgraph import KnowledgeGraph
from qwen_model import generate_answer


class GraphRAGApp:
    def __init__(self, emb_path, csv_path, model_name="sentence-transformers/all-MiniLM-L6-v2", device='cpu'):
        # Load embeddings
        data = torch.load(emb_path)
        self.x = data["x"]          # semantic embeddings
        self.h = data["h"]          # structural embeddings
        self.node2id = data["node2id"]

        # Reverse mapping
        self.id2node = {v: k for k, v in self.node2id.items()}

        # Load KG
        df = pd.read_csv(csv_path, names=["Subject", "Predicate", "Object"])
        self.kg = KnowledgeGraph(df, self.node2id)
        self.kg.build_graph()
        self.kg.build_adjacency()

        # Load embedding model (for query), embedding dimension is 384 for all-MiniLM-L6-v2 model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # ---------- QUERY EMBEDDING ----------
    def embed_query(self, query):
        "Helper function to embed the query into a [1, 384] vector"
        encoded = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)

        pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        pooled = F.normalize(pooled, dim=1)

        return pooled.cpu()  # [1, 384]

    # ---------- HYBRID RETRIEVAL ----------
    def retrieve(self, query, top_k=10, alpha=0.8):
        """
        Helper function to embed the query and retrieve context-relevant nodes\n
        This is equivalent implementation to the subclass of LlamaIndex's BaseRetrieval

        Parameters:
            + query: User query sentence
            + top_k: K most relevant nodes
            + alpha: Weighting for the semantic similarity 
        """
        
        q_emb = self.embed_query(query)  # [1, 384]

        # Compute similarities
        sim_sem = F.cosine_similarity(q_emb, self.x)
        sim_struct = F.cosine_similarity(q_emb, self.h)

        # Hybrid score
        scores = alpha * sim_sem + (1 - alpha) * sim_struct

        topk = torch.topk(scores, k=top_k)
        node_ids = topk.indices.tolist()

        nodes = [self.id2node[i] for i in node_ids]

        # Remove long phrases
        return [
            n for n in nodes
            if 1 <= len(n.split()) <= 3  
        ]


    # ---------- CONTEXT BUILD ----------
    def build_context(self, nodes, max_triples=30):
        """
        Build KG-based context from retrieved nodes
        """

        triples = []

        for node in nodes:
            neighbors = self.kg.get_neighbors(node)
            triples.extend(neighbors)

        # Deduplicate
        triples = list(set(triples))[:max_triples]

        # Convert to text
        context = []
        for s, p, o in triples:
            context.append(
                f"{s.replace('_',' ')} {p.replace('_',' ')} {o.replace('_',' ')}"
            )

        return "\n".join(context)

    # ---------- FULL PIPELINE ----------
    def answer(self, query):
        """
        Main function to response a user query
        """
        
        nodes = self.retrieve(query)
        context = self.build_context(nodes)

        # -----For debugging-----

        # print("===" * 30)
        # print("Retrieved:")
        # print("Nodes:")
        # print(nodes)
        # print("---" * 30)
        # print("Context:")
        # print(context)
        # print("===" * 30)

        prompt = f"""You are given facts from a knowledge graph.

        Task:
        - Think step by step using the facts.
        - You MUST complete your reasoning.
        - Then you MUST provide the final answer.

        Output format (strict):
        <think>
        your reasoning
        </think>
        <answer>
        one short sentence answer
        </answer>

        Rules:
        - Always close both tags.
        - The answer MUST appear inside <answer>.
        - Do not stop early.
        - Do not omit the answer.

        Context:
        {context}

        Question: {query}
        """

        raw_answer = generate_answer(prompt)
        # print("Raw answer:")
        # print(raw_answer)

        return self.clean_output(raw_answer)

    def clean_output(self, text: str) -> str:
        # Case 1: proper <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Case 2: <answer> exists but no closing tag
        if "<answer>" in text:
            return text.split("<answer>")[-1].strip().split("\n")[0]

        # Case 3: fallback to "Final Answer:"
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip().split("\n")[0]

        # Case 4: fallback to last non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[-1] if lines else "I don't know."