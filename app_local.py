# For testing the LLM Q&A service

from online.graph_rag import GraphRAGApp

if __name__ == "__main__":
    app = GraphRAGApp(
        emb_path="./artifacts/embeddings/embs_epochs200_lr0.0001_weight_decay0.0001_hidden_dim128_num_layer3_neg_k5_margin0.2_lambda_align0.03.pt",
        csv_path="./artifacts/graph_triplets/graph_edges.csv",
        topk=6,
        semantic_contribute=0.5,
        score_tolerance_factor=0.4,
        verbose=True
    )

    while True:
        query = input("\nQuery: ")
        stream = app.answer(query)
   
        for chunk in stream:
            try:
                content = chunk['choices'][0]['text']
                if content:
                    print(content, end="", flush=True)
            except (KeyError, IndexError):
                continue
       