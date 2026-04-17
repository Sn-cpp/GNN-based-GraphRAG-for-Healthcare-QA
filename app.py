import gradio as gr
from graph_rag import GraphRAGApp

# Initialize app (load once)
app = GraphRAGApp(
    emb_path="./artifacts/embeddings/embs_epochs200_lr0.0001_hidden_dim256_num_layer3_weight_decay0.0001.pt",
    csv_path="./artifacts/graph_triplets/graph_edges.csv"
)

# Inference function
def answer_query(query):
    if not query or query.strip() == "":
        return "Please enter a valid question."
    
    try:
        response = app.answer(query)
        return response if response else "No answer found."
    except Exception as e:
        return f"Error: {str(e)}"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# GNN-based GraphRAG for Healthcare QA")
    gr.Markdown("Ask a healthcare-related question. The system retrieves knowledge from a graph and generates an answer.")

    with gr.Row():
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is anatomy?"
        )

        gr.Examples(
        examples=[
                "What is anatomy?",
                "What is aspirin?",
                "Is heart a part of cardiovascular system?"
            ],
            inputs=query_input
        )
        
    with gr.Row():
        submit_btn = gr.Button("Submit")

    with gr.Row():
        output = gr.Textbox(
            label="Answer",
            lines=3
        )

    submit_btn.click(fn=answer_query, inputs=query_input, outputs=output)

# Launch
if __name__ == "__main__":
    demo.launch()