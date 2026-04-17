from llama_cpp import Llama


class QwenLLM:
    def __init__(self, repo_id="Jackrong/Qwen3.5-4B-Neo-GGUF", filename="Qwen3.5-4B.Q4_K_M.gguf", n_ctx=8192, n_threads=2, n_gpu_layers=0):
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    def generate(self, prompt):
        output = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.3,
            top_p=0.9
        )

        return output["choices"][0]["text"].strip()


# Singleton (simple)
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = QwenLLM()
    return _llm


def generate_answer(prompt):
    llm = get_llm()
    return llm.generate(prompt)