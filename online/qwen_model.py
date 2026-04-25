from llama_cpp import Llama


class QwenLLM:
    _instance = None
    """
    Wrapper class to initialize Qwen Large Language Model
    """
    
    def __init__(self, repo_id="Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF", filename="Qwen3.5-4B.Q5_K_M.gguf", n_ctx=6144, n_threads=2, n_gpu_layers=0):
        if QwenLLM._instance is None:
            QwenLLM._instance = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
        self.llm = QwenLLM._instance
    
    def generate(self, prompt: str, stream=True):
        """
        Function to perform LLM reasoning and answering
        """
        
        output = self.llm(
            prompt,
            stream=stream,
            max_tokens=4096,
            temperature = 0.2,
            top_p = 0.9,
            stop=["</answer>"]
        )
        
        return output