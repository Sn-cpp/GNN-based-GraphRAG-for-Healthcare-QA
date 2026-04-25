import argparse
import os


if os.environ.get("FORCE_CLI_TQDM") == "1":
    from tqdm import tqdm
else:
    from tqdm.autonotebook import tqdm

import torch

from datasets import load_dataset

from offline.text_processor import TextProcessor
from offline.triplets_extractor import RebelExtractor
from offline.triplets_validator import DeBERTa_Validator

def ingest(override_args: argparse.Namespace):
    def process_text(text_chunk: str, sample_idx: int):
        """
        Pipeline to produce triplets from a paragraph
        """

        samplebar.set_description_str("[Splitting sentences...]") 
        sentences = text_processor.split_sentences(text_chunk)
        if len(sentences) < 1:
            return []

        rebel_inputs = [sentences[0]]
        for i in range(1, len(sentences)):
            rebel_inputs.extend([
                sentences[i-1] + " " + sentences[i],
                sentences[i]
            ])


        # Utilize REBEL seq2seq model to extract triplets
        triplets = []
        samplebar.set_description_str("[Extracting triplets...]")
        for sent_idx, s in  enumerate(tqdm(rebel_inputs, position=1, desc="Samples", leave=False, disable=os.environ.get("TQDM_DISABLE") == "1")):
            try:
                parsed = triplets_extractor.parse_output(triplets_extractor.extract(s))
                triplets.extend(parsed)
            except Exception as e:
                tqdm.write(f"[WARN] Failed on sentence no. {sent_idx+1} of sample no. {sample_idx+1}: {e}")
                continue

        triplets = list(set(triplets))
        if len(triplets) < 1:
            return []
 

        # Utilize DeBERTa to validate triplets using the text as context
        samplebar.set_description_str("[Validating triplets with DeBERTa...]")
        valid_triplets = []
        try:
            valid_triplets = triplets_validator.validate_triplets(triplets, text_chunk)
        except Exception as e:
            tqdm.write(f"[WARN] Validation of sample no. {sent_idx+1} failed: {e}")
            return [] # Return empty instead of un-validated facts


        # Apply normalization rules (lower-case, replace Greek characters, no weird spaces,...)
        # The use spaCy to lemmatize entites and simplify relation phrases
        refined_triplets = []
        for s, r, o in valid_triplets:
            try:
                p_s = text_processor.normalize_text(text_processor.lemmatize(s))
                p_r = text_processor.normalize_text(text_processor.lemmatize(text_processor.simplify_phrase(r)))
                p_o = text_processor.normalize_text(text_processor.lemmatize(o))
                refined_triplets.append((p_s, p_r, p_o))
            except:
                tqdm.write(f"[WARN] Triplets refining failed: ({s}, {r}, {o})")
                continue
        
        return refined_triplets

    # Allowed categories
    allowed_categories = ['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']

    default_args = argparse.Namespace(
        data_path="cogbuji/medqa_corpus_en",
        category=allowed_categories,
        num_samples=100,
        start_idx=0,
        output_path='./artifacts/raw_triplets',
        cuda=False,
        cuda_extractor=False
    )

    args = argparse.Namespace(**{**vars(default_args), **vars(override_args)})

    # Safe-guard
    if args.start_idx < 0:
        print("Invalid starting index. Fallback to 0")
        args.start_idx = 0 

    args.category = eval(args.category)
    
    invalid_category = set(args.category) - set(allowed_categories)
    if len(invalid_category):
        print("Detecting invalid category:", ", ".join(invalid_category))
        print("Allowed category:", ", ".join(allowed_categories))
        return


    # Make directory for outputs
    os.makedirs(args.output_path, exist_ok=True)
    
    # GPU mode checker
    rebel_device = torch.device("cuda:0" if args.cuda_extractor and torch.cuda.is_available() else "cpu")  
    reberta_device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
   
    text_processor = TextProcessor()
    print("spaCy loaded.")

    triplets_extractor = RebelExtractor(device=rebel_device)
    print("REBEL loaded.")

    triplets_validator = DeBERTa_Validator(device=reberta_device)
    print("DeBERTa loaded.")
    
    
    for ctg in args.category:
        total_triplets = 0
        print(f"Category: {ctg}")

        # Load dataset with the specified category
        dataset = load_dataset(
            args.data_path,
            name=ctg,
            split='train',
            trust_remote_code=True
        )
        dataset_len = len(dataset)

        # Handle starting index out of index error
        if (args.start_idx >= dataset_len):
            tqdm.write(f"[WARN] Out of index for category [{ctg}]: starting index (start_idx) must be smaller than {dataset_len}. Skipping this category...")
            continue

        # Handle number of samples: automatically truncated to len(dataset)
        num_samples = dataset_len - args.start_idx  
        if args.num_samples < num_samples:
            num_samples = args.num_samples
        else:
            tqdm.write(f"[WARN] Out of index: Automatically reducing num_samples from {args.num_samples} reduced to {num_samples}")

        # Create progress bar
        samplebar = tqdm(range(num_samples), desc="Samples", position=0, disable=os.environ.get("TQDM_DISABLE") == "1")

         # Reset existing(if) files
        open(os.path.join(args.output_path, f'{ctg}_numsamples{args.num_samples}_start{args.start_idx}.csv'), 'w').close()

        for i in samplebar:
            # Free GPU every 20 epochs
            if (i+1) % 20 == 0: 
                torch.cuda.empty_cache()
            
            # Replacement when bars are disabled
            if os.environ.get("TQDM_DISABLE") == "1":
                if (i+1) % 10 == 0:
                    tqdm.write(f"Sample {i+1}/{num_samples}")

            try:
                # Call the pipeline
                triplets = process_text(dataset[args.start_idx + i]['text'], i+1)
                total_triplets += len(triplets)

                # Update information
                samplebar.set_postfix({"Total num triplets" : total_triplets })
                
                # Append triplets to file
                with open(os.path.join(args.output_path, f'{ctg}_numsamples{args.num_samples}_start{args.start_idx}.csv'), 'a', encoding="utf-8") as f:
                    for s, r, o in triplets:
                        f.write(f'{s},{r},{o}\n')
            
            except Exception as e:
                tqdm.write(f"Sample no.{i+1} failed: {e}")
                continue

        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="cogbuji/medqa_corpus_en", help="Hugging Face Repository path of the dataset, set to the local path if needed")
    parser.add_argument('--category', type=str, default="['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']", help="Specify which categories to be used. Allowed category: core_clinical, basic_biology, pharmacology, psychiatry")
    parser.add_argument('--cuda', action='store_true', help="Use GPU for LLM refining. Require CUDA-compiled llama-cpp-python")
    parser.add_argument('--cuda_extractor', action='store_true', help="Use GPU for triplets extractor (REBEL)")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples in a category to be used")
    parser.add_argument('--output_path', type=str, default='./artifacts/raw_triplets', help="Output directory for triplets CSV files")
    parser.add_argument('--start_idx', type=int, default=0, help="Starting sample index")

    args = parser.parse_args()
    ingest(args)