from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
import pandas as pd

if os.environ.get("FORCE_CLI_TQDM") == "1":
    from tqdm import tqdm
else:
    from tqdm.autonotebook import tqdm


class DeBERTa_Validator:
    """
    Utilize DeBERTa-v3-base-mnli-fever-anli to validate triplets extracted from a paragraph
    """
    
    def __init__(self, model_id="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=torch.device('cpu')):
        self.device = device

        # Initialize the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./hf_cache")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            cache_dir="./hf_cache"
        ).to(self.device)
        self.model.eval()

        # Indices of scoring labels: 
        self.scorings = { v:k for k, v in self.model.config.id2label.items() }

    def score_triplets(self, triplets: list[tuple, ], paragraph: str):
        """
        Function to score the support of triplets given a paragraph as context
        """
        
        # Construct triplets into sentences as hypotheses for the model
        hypotheses = [f"{s} {r} {o}." for s, r, o in triplets]

        # Tokenize and inference
        inputs = self.tokenizer(
            [paragraph] * len(hypotheses),
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
       
        results = probs.cpu().numpy()

        del inputs, logits, probs

        return results
    
    def validate_triplets(self, triplets: list[tuple, ], paragraph: str,
            entail_w    = 1.0,
            neutral_w   = 1.0,
            contra_w    = 0.5,
            min_entail  = 0.97,
            max_neutral = 0.1,
            max_contra  = 0.01,
            min_score   = 0.88
        ):
        """
        Function to validate triplets given a paragraph as context.\n
        Return valid triplets satisfying the minimum threshold computed by a weighted combination of:
        - Entailment: stated or clearly implied.
        - Neutral: might be true, but it is not stated.
        - Contradiction: denied by the paragraph.
        """

        pbar = tqdm(total=1, desc="[Validating...]", position=1, leave=False, disable=os.environ.get("TQDM_DISABLE") == "1")

        # 
        triplets_df = pd.DataFrame(triplets, columns=["S", "R", "O"]
        )

        # Compute the Entailment, Neutral and Contradiction
        probs = self.score_triplets(triplets, paragraph)

        ent = probs[:, self.scorings['entailment']]
        contr= probs[:, self.scorings['contradiction']]
        neu = probs[:, self.scorings['neutral']]
        # Weighted combination for metrics
        score = entail_w*ent - neutral_w*neu - contra_w * contr

        # Get valid triplets
        mask = (ent >= min_entail) & (neu <= max_neutral) & (contr <= max_contra) & (score >= min_score)
        valid_triplets = triplets_df.iloc[mask]

        pbar.update(1)
        return [(row.S, row.R, row.O) for row in valid_triplets.itertuples()]









        

