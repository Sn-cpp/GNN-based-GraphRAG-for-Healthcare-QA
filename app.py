import gradio as gr
from online.graph_rag import GraphRAGApp

# Initialize backend (once)
app = GraphRAGApp(
    emb_path="./artifacts/embeddings/embs_epochs200_lr0.0001_weight_decay0.0001_hidden_dim128_num_layer3_neg_k5_margin0.2_lambda_align0.03.pt",
    csv_path="./artifacts/graph_triplets/graph_edges.csv",
    topk=6,
    semantic_contribute=0.5,
    score_tolerance_factor=0.4
)

# Streaming inference logic
def answer_query(query):
    if not query or query.strip() == "":
        yield "Please enter a valid question."
        return

    # Immediate UI feedback
    yield "Thinking...", gr.update(interactive=False)
    REASONING_START_TAG = "<think>" 
    START_TAG = "</think>\n\n<answer>"

    # Store stream outputs from LLM with buffer, and start displaying result after the signal </think>\n\n<answer>
    buffer = ""
    start_yielding = False
    has_think_tag = True

    stream_output = app.answer(query, stream=True) #Streaming LLM output
    for raw_chunk in stream_output:
        chunk = raw_chunk["choices"][0]["text"]
        buffer += chunk

        # Handle when LLM didn't return <think>...</think> block
        if has_think_tag:
            if len(buffer) > 2 * len(REASONING_START_TAG) and REASONING_START_TAG not in buffer:
                start_yielding = True
                has_think_tag = False

        # Un-comment this for debugging
        print(chunk, end="", flush=True)

        if start_yielding:
            yield buffer, gr.update(interactive=False)
            continue
   
        idx = buffer.find(START_TAG, len(buffer) // 2)
        if idx != -1:
            buffer = buffer[idx+len(START_TAG):]
            start_yielding = True

    if len(buffer) == 0:
        buffer = "I don't know"

    yield buffer, gr.update(interactive=True)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# GNN-based GraphRAG for Healthcare QA")
    gr.Markdown(
        "Ask a healthcare-related question. "
        "The system retrieves knowledge from a graph and generates an answer."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is anatomy?"
        )

        gr.Examples(
            examples=[
                "What is anatomy ?",
                "Is heart a part of cardiovascular system ?",
                "What is related to the central nervous system ?"
            ],
            inputs=query_input
        )

    with gr.Row():
        submit_btn = gr.Button("Submit")

    with gr.Row():
        output = gr.Textbox(
            label="Answer",
            lines=5
        )

    # Streaming happens automatically because answer_query uses `yield`
    submit_btn.click(
        fn=answer_query,
        inputs=query_input,
        outputs=[output, submit_btn]
    )

# Launch
if __name__ == "__main__":
    demo.launch()