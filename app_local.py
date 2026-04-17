from graph_rag import GraphRAGApp

if __name__ == "__main__":
    app = GraphRAGApp(
        emb_path="./artifacts/embeddings/embs_epochs200_lr0.0001_hidden_dim256_num_layer3_weight_decay0.0001.pt",
        csv_path="./artifacts/graph_triplets/graph_edges.csv"
    )

    while True:
        query = input("Query: ")
        answer = app.answer(query)
        print("\nAnswer:\n", answer)

    # answer = app.answer("Is hemoglobin a protein?")
    # print("\nAnswer:\n", answer)