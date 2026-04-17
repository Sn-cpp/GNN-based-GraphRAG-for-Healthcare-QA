from llama_cpp import Llama
import re
from tqdm import tqdm

from models.mistral_prompt import MistralPrompt

class MistralRefiner:
    """
    Utilize Mistral 7B Instruct LLM to refine triplets produced by the REBEL model
    """
    def __init__(self,
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            context_size=4096,
            verbose=False,
            num_cpu_threads=8,
            use_gpu=False
        ):
        
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=context_size,
            verbose=verbose,
            n_threads=num_cpu_threads, 
            n_gpu_layers=-1 if use_gpu else 0
        )

    def send_prompt(self, prompt: str) -> str:
        """
        Send prompt to the LLM 
        """

        # Call LLM with prompt and parameters
        output = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            repeat_penalty=1.1,
            stop=["</s>"]
        )

        return output["choices"][0]["text"]
    
    def parse_triplets(self, text: str):
        """
        Helper function to parse triplets from LLM return
        """
        
        triplets = []
        # Locate (A, r, B) pattern
        matches = re.findall(r"\((.*?),\s*(.*?),\s*(.*?)\)", text)

        if matches:
            base = matches
        else:
            # Fallback to chain format if failed
            tokens = [t.strip() for t in text.split(",") if t.strip()]
            base = []

            if len(tokens) >= 3 and len(tokens) % 2 == 1:
                for i in range(0, len(tokens) - 2, 2):
                    base.append((tokens[i], tokens[i+1], tokens[i+2]))

        # Handle multi-values objects
        for s, r, o in base:

            subjects = re.split(r",| and ", s)
            objects = re.split(r",| and ", o)

            for ss in subjects:
                for oo in objects:
                    s_clean = self.normalize_text(ss)
                    r_clean = self.normalize_text(r)
                    o_clean = self.normalize_text(oo)

                    if self.validate_triplet(s_clean, r_clean, o_clean):
                        triplets.append((s_clean, r_clean, o_clean))

        return triplets

    def refine_triplets(self, triplets: list[tuple, ]):
        """
        Helper function to refine triplets with Mistral AI in batches
        """

        # Define number of triplets per batch 
        BATCH_SIZE = 64
        all_results = []
        for i in tqdm(range(0, len(triplets), BATCH_SIZE), desc='Batches', leave=False, position=1):
            
            # Slice triplets into batch
            batch = triplets[i:i+BATCH_SIZE]

            # Build the prompt
            prompt = MistralPrompt.get_refining_prompt(batch)
            
            # Send prompt
            raw = self.send_prompt(prompt)

            # Extract results 
            data = self.parse_triplets(raw)

            # Fallback: use REBEL results 
            if len(data) == 0:
                for s, r, o in batch:
                    s_clean = self.normalize_text(s)
                    r_clean = self.normalize_text(r)
                    o_clean = self.normalize_text(o)

                    if self.validate_triplet(s_clean, r_clean, o_clean):
                        all_results.append((s_clean, r_clean, o_clean))
            else:
                all_results.extend(data)

        return list(set(all_results))
            
    
    def cleanup(self):
        if self.llm is not None:
            self.llm.close()
            self.llm = None

    def validate_triplet(self, s, r, o):
        if s == o:
            return False
        if not s or not r or not o:
            return False
        return True
    
    def normalize_text(self, x):
            if x is None:
                return ""

            x = str(x)

            # remove backslashes
            x = x.replace("\\", "")

            # normalize case and spaces
            x = x.strip().lower()

            return x