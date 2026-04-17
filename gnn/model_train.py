import argparse
import numpy as np
import torch
import random
import os

from tqdm.autonotebook import tqdm

from embeddings_init import EmbeddingInitializer
from kgraph import KnowledgeGraph
from r_gcn import R_GCN


def train_model(override_args: argparse.Namespace):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    def sample_negative(src, num_nodes, device, generator):
        # Function to artifically generate fake edges
        neg_src = src
        neg_dst = torch.randint(0, num_nodes, src.shape, device=device, generator=generator)
        return neg_src, neg_dst
    


    default_args = argparse.Namespace(
        data_file="graph_edges.csv",
        data_path="../artifacts/graph_triplets",
        output_path="../artifacts/embeddings",
        cuda=False,
        epochs=200,
        lr=1e-4,
        weight_decay=1e-4,
        hidden_dim=256,
        num_layer=3
    )

    args = argparse.Namespace(**{**vars(default_args), **vars(override_args)})

    print("===" * 20)
    print("Arguments: ", args)
    print("===" * 20)

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    # Set generator's seed for negative edges sampling
    g = torch.Generator(device=device)
    g.manual_seed(seed)


    # ---Data loading and embeddings making---
    embedd_initializer = EmbeddingInitializer()
    data = embedd_initializer.run(path=os.path.join(args.data_path, args.data_file))

    print("Data and Embeddings done.")


    x = data["x"].to(device)
    node2id = data["node2id"]
    df = data["df"]

    # Build graph using KnowledgeGraph class
    kg = KnowledgeGraph(df, node2id=node2id)
    kg.build_graph()

    # Retrieve graph data
    graph_data = kg.get_data()
    edge_index = graph_data['edge_index'].to(device)
    edge_type = graph_data['edge_type'].to(device)
    rel2id = graph_data['rel2id']
    
    num_nodes = x.size(0)
    num_relations = len(rel2id)

    # Initialize model
    model = R_GCN(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=384, # Fix the output embedddings size to 384 to sync with the later query embedder
        num_relations=num_relations,
        num_layers=args.num_layer
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # List of sources and tails
    src = edge_index[0]
    dst = edge_index[1]

    for epoch in tqdm(range(args.epochs), position=0, desc="Epoch"):
        model.train()
        optimizer.zero_grad()

        # Forward
        h = model(x, edge_index, edge_type)

        # Positive edges
        pos_score = model.score(h, src, dst)

        # Negative edges
        neg_src, neg_dst = sample_negative(src, num_nodes, device, generator=g)
        neg_score = model.score(h, neg_src, neg_dst)

        # Labels
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_labels, neg_labels])

        # Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

        tqdm.write(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")


    save_dict = {
        "x": x.detach().cpu(),
        "h": h.detach().cpu(),        # [num_nodes, out_dim]
        "node2id": node2id,
        "rel2id": rel2id
    }

    # Ensure directory exists
    os.makedirs(args.output_path, exist_ok=True)

    saved_path = os.path.join(args.output_path,
        f"embs_epochs{args.epochs}_lr{args.lr}_hidden_dim{args.hidden_dim}_num_layer{args.num_layer}_weight_decay{args.weight_decay}.pt"
    )

    torch.save(save_dict, saved_path)

    tqdm.write(f"Saved embeddings to: {saved_path}")

    return model, h, rel2id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="graph_edges.csv", help="Name of the file containing triplets")
    parser.add_argument("--data_path", type=str, default="../artifacts/graph_triplets", help="Directory containing triplets file")
    parser.add_argument("--output_path", type=str, default="../artifacts/embeddings", help="Directory for the final embeddings")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA device")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--num_layer", type=int, default=3, help="Number of hidden layer(s)")

    args = parser.parse_args()
    train_model(args)