import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

class RebelExtractor:
    """
    Utilize REBEL model to extract triplet (subject, relation, object) from sentence
    """
    def __init__(self, model_name="Babelscape/rebel-large", device='cpu'):
        self.device = device
        
        # Load the Rebel model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./hf_cache')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./hf_cache').to(self.device)

    def extract(self, sentence: str):
        """
        Forward function, recognize \\<subj>, \\<obj> elements in the given sentence
        """
        
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model.generate(
            **encoded,            
            max_length=256,       
            num_beams=4,         
            do_sample=False
        )

        decoded = self.tokenizer.decode(
            outputs[0].cpu(),
            skip_special_tokens=False
        )

        return decoded
    
    def parse_output(self, text: str) -> list[tuple]:
        """
        Parse REBEL output into (subject, relation, object)
        """

        def clean(x: str) -> str:
            return (
                x.replace("</s>", "")
                .replace("<s>", "")
                .strip()
                .lower()
            )

        triplets = []

        # Tokenize by special markers while keeping them
        tokens = re.split(r"(<triplet>|<subj>|<obj>)", text)

        subj, obj, rel = None, None, None
        state = None  # tracks what we are reading

        for token in tokens:
            token = token.strip()

            if token == "<triplet>":
                # flush previous triplet if complete
                if subj and rel and obj:
                    triplets.append((clean(subj), clean(rel), clean(obj)))
                subj, obj, rel = None, None, None
                state = "triplet"

            elif token == "<subj>":
                state = "subj"

            elif token == "<obj>":
                state = "obj"

            else:
                if not token:
                    continue

                if state == "triplet":
                    subj = token

                elif state == "subj":
                    obj = token

                elif state == "obj":
                    rel = token

        # flush last triplet
        if subj and rel and obj:
            triplets.append((clean(subj), clean(rel), clean(obj)))

        return triplets