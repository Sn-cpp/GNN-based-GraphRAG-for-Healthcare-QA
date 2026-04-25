import argparse
import numpy as np
import torch
import random
import os
import torch.nn.functional as F

if os.environ.get("FORCE_CLI_TQDM") == "1":
    from tqdm import tqdm
else:
    from tqdm.autonotebook import tqdm
    
from cores.kgraph.knowledge_graph import KnowledgeGraph
from cores.gnn.r_gcn import R_GCN
from cores.embedder.embeddings_init import TextEmbedder


def train_model(override_args: argparse.Namespace):
    # Reproduction config
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Arguments re-solving
    default_args = argparse.Namespace(
        data_file="graph_edges.csv",
        data_path="./artifacts/graph_triplets",
        output_path="./artifacts/embeddings",
        cuda=False,
        cuda_embedder=False,
        epochs=200,
        lr=1e-4,
        weight_decay=1e-4,
        hidden_dim=128,
        num_layer=2,
        neg_k=5,
        margin=0.2,
        lambda_align=0.03
    )

    args = argparse.Namespace(**{**vars(default_args), **vars(override_args)})

    print("Arguments: ", args)

    gnn_device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    embedder_device = 'cuda:0' if args.cuda_embedder and torch.cuda.is_available() else 'cpu'

    g = torch.Generator(device=gnn_device)
    g.manual_seed(seed)

    # Use all-MiniLM-L6-v2 to create semantic embeddings 
    text_embedder = TextEmbedder(device=embedder_device)
    data = text_embedder.run(path=os.path.join(args.data_path, args.data_file))

    print("Embedding done.")

    h_text = data["h_text"].to(gnn_device)   # raw semantic embeddings 
    node2id = data["node2id"]
    df = data["df"]

    # Build knowledge graph

    kg = KnowledgeGraph(df, node2id=node2id)
    kg.build_graph()
    graph_data = kg.get_data()

    edge_index = graph_data['edge_index'].to(gnn_device)
    edge_type = graph_data['edge_type'].to(gnn_device)
    rel2id = graph_data['rel2id']

    num_nodes = h_text.size(0)
    num_relations = graph_data['num_relations']

    # Define R_GCN Model
    model = R_GCN(
        in_dim=h_text.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=384,
        num_relations=num_relations,
        num_layers=args.num_layer
    ).to(gnn_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    src = edge_index[0]
    dst = edge_index[1]
    rel = edge_type

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        model.train()
        optimizer.zero_grad()

        # Forward
        h_struct = model(h_text, edge_index, edge_type)  # Apply model to find structural embeddings

        # Normalize both textural and structural spaces
        h_struct = F.normalize(h_struct, dim=1)
        h_text_norm = F.normalize(h_text, dim=1)

        # Use node contexts as query, project it into the structural space
        q_text = h_text_norm[src]                     
        q_struct = model.project_query(q_text) # project into structural space
        q_struct = F.normalize(q_struct, dim=1)

        # Positive score
        pos_vec = h_struct[dst]
        pos_score = F.cosine_similarity(q_struct, pos_vec, dim=1)

        # Negative sampling (multi-negative)
        neg_dst = torch.randint(0, num_nodes, (dst.size(0), args.neg_k),
                                device=gnn_device, generator=g)

        # Negative score
        neg_vec = h_struct[neg_dst]             
        q_expand = q_struct.unsqueeze(1)        
        neg_score = F.cosine_similarity(q_expand, neg_vec, dim=2)

        # Contrastive loss
        pos_score = pos_score.unsqueeze(1)
        loss_contrast = torch.mean(F.relu(args.margin - pos_score + neg_score))

        # Apply soft semantic alignment with cosine similarity
        h_text_detached = h_text_norm.detach()
        align = F.cosine_similarity(h_struct, h_text_detached, dim=1) 
        loss_align = torch.mean((1 - align))

        # Use variance preservation method to prevent Structural embeddings from collapsing into Semantic embeddings
        std = torch.std(h_struct, dim=0)
        loss_var = torch.mean(F.relu(1 - std))

        # Compute loss
        loss = (
            loss_contrast
            + args.lambda_align * loss_align      # reuse lambda_align as align weight
            + 0.01 * loss_var                    # small stabilizer
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            corr = (h_struct * h_text_norm).sum(dim=1).mean().item()
            tqdm.write(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Semantic-Structural Correlation: {corr:.6f}")

    # Save embeddings to checkpoint file
    save_dict = {
        "query_proj": {k: v.cpu() for k, v in model.query_proj.state_dict().items()}, # query projection, (move all components to cpu before saving)
        "h_struct": F.normalize(h_struct.detach().cpu()),   # structural embedding
        "h_text": F.normalize(h_text.detach().cpu()),       # raw embedding
        "node2id": node2id,
        "rel2id": rel2id
    }

    os.makedirs(args.output_path, exist_ok=True)

    path = os.path.join(
        args.output_path,
         f"embs_epochs{args.epochs}_lr{args.lr}_weight_decay{args.weight_decay}_hidden_dim{args.hidden_dim}_num_layer{args.num_layer}_neg_k{args.neg_k}_margin{args.margin}_lambda_align{args.lambda_align}.pt"
    )

    torch.save(save_dict, path)
    print(f"Saved to {path}")

    return model, save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="graph_edges.csv", help="Name of the file containing triplets")
    parser.add_argument("--data_path", type=str, default="./artifacts/graph_triplets", help="Directory containing triplets file")
    parser.add_argument("--output_path", type=str, default="./artifacts/embeddings", help="Directory for the final embeddings")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA device for training model")
    parser.add_argument("--cuda_embedder", action="store_true", help="Use CUDA device for nodes context embedding")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--num_layer", type=int, default=2, help="Number of hidden layer(s)")
    parser.add_argument("--neg_k", type=int, default=5, help="Number of negative edge sampled")
    parser.add_argument("--margin", type=float, default=0.2, help="Separation constraint between positives and negatives of constrastive objective")
    parser.add_argument("--lambda_align", type=float, default=0.03, help="Semantic alignment weight between semantic and structural embeddings")

    args = parser.parse_args()
    train_model(args)