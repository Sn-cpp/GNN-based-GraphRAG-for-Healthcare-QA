from tqdm import tqdm
import argparse
import os

from datasets import load_dataset

from models.sentences_producer import SentenceProducer
from models.triplets_extractor import RebelExtractor
from models.triplets_refiner import MistralRefiner

def ingest(override_args: argparse.Namespace):
    def process_text(text: str):
        """
        Pipeline to produce triplets from a paragraph
        """
        
        # Produce sentences from paragraph
        samplebar.set_description_str("[Splitting sentences...]")
        sentences = sentence_parser.split_sentences(text)


        # Utilize REBEL seq2seq model to extract triplets
        triplets = []
        samplebar.set_description_str("[Extracting triplets per sentence...]")
        for sent in tqdm(sentences, position=1, desc="Sentences", leave=False):
            raw_output = triplets_extractor.extract(sent)
            parsed = triplets_extractor.parse_output(raw_output)
            triplets.extend(parsed)


        # Use Mistral to normalize triplets
        samplebar.set_description_str("[Refining triplets with Mistral...]")
        refined_triplets = triplets_refiner.refine_triplets(triplets)

        return refined_triplets



    # Allowed categories
    allowed_categories = ['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']

    default_args = argparse.Namespace(
        data_path="cogbuji/medqa_corpus_en",
        category=allowed_categories,
        num_samples=100,
        start_idx=0,
        output_path='../artifacts/raw_triplets',
        cuda=False
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
    
   
    sentence_parser = SentenceProducer()
    triplets_extractor = RebelExtractor()
    triplets_refiner = MistralRefiner(use_gpu=args.cuda)

    
    for ctg in args.category:
        total_triplets = 0
        tqdm.write(f"Category: {ctg}")
        samplebar = tqdm(range(args.num_samples), desc="Samples", position=0)

        dataset = load_dataset(
            args.data_path,
            name=ctg,
            split='train',
            trust_remote_code=True
        )

         # Reset existing(if) files
        open(os.path.join(args.output_path, f'{ctg}_numsamples{args.num_samples}_start{args.start_idx}.csv'), 'w').close()

        for i in samplebar:
            if args.start_idx + i >= len(dataset):
                tqdm.write("Out of samples. Exitting...")
                return
            
            try:
                triplets = process_text(dataset[args.start_idx + i]['text'])
                total_triplets += len(triplets)

                samplebar.set_postfix({"Total num triplets" : total_triplets })
                with open(os.path.join(args.output_path, f'{ctg}_numsamples{args.num_samples}_start{args.start_idx}.csv'), 'a') as f:
                    for s, r, o in triplets:
                        f.write(f'{s},{r},{o}\n')

            except Exception as e:
                tqdm.write(f"Sample no.{i+1} failed. Skipping...")
                continue


    #Explicitly free the LLM
    triplets_refiner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="cogbuji/medqa_corpus_en", help="Hugging Face Repository path of the dataset, set to the local path if needed")
    parser.add_argument('--category', type=str, default="['core_clinical', 'basic_biology', 'pharmacology', 'psychiatry']", help="Specify which categories to be used. Allowed category: core_clinical, basic_biology, pharmacology, psychiatry")
    parser.add_argument('--cuda', action='store_true', help="Use GPU for LLM refining. Require CUDA-compiled llama-cpp-python")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples in a category to be used")
    parser.add_argument('--output_path', type=str, default='../artifacts/raw_triplets', help="Output directory for triplets CSV files")
    parser.add_argument('--start_idx', type=int, default=0, help="Starting sample index")

    args = parser.parse_args()
    ingest(args)