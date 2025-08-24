import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaTokenizer
import nltk
from nltk import tokenize
import os
import sys
from tqdm import tqdm
import warnings
import tensorflow_hub as hub
import tensorflow as tf
from dotenv import load_dotenv
from accelerate import Accelerator, InitProcessGroupKwargs
from argparse import Namespace
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv() 
token = os.getenv("HF_TOKEN")  

from huggingface_hub import login
if token:
    login(token=token)

LLM_MODELS = [
    # LLMs
    ('meta-llama/Llama-3.2-1B', 'llama3.2'),
    ('google/gemma-3-1b-it', 'gemma'),
    ('mistralai/Mistral-7B-v0.1', 'mistral'),
    ("apple/OpenELM-270M", "openelm"),
    ("allenai/OLMo-2-0425-1B", "olmo"),
    ("Qwen/Qwen3-0.6B", "qwen"),
    ('AtlaAI/Selene-1-Mini-Llama-3.1-8B', 'selene-llama'),
    ('tiiuae/falcon-7B', 'falcon'),
    ('tiiuae/Falcon3-7B-Base', 'falcon3'),
    ('yulan-team/YuLan-Mini', 'yulan-mini'),

    # Classical models
    ('sentence-transformers/all-MiniLM-L6-v2', 'sbert-mini'),
    ('sentence-transformers/all-mpnet-base-v2', 'sbert-l'),
    ('universal-sentence-encoder', 'use'),
    ('roberta-base', 'roberta'),
    ('princeton-nlp/sup-simcse-roberta-base', 'simcse'),    

    # New models
    ('DILAB-HYU/SentiCSE', 'senticse'),
    ('geneol','geneol')
]

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration constants
LAMBDA_PARAM = 0.3 # optimal
SUMMARY_TYPE = "Abstractive"  # Fixed to Abstractive summaries only

class NDCGAbsEvaluator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.lambda_param = LAMBDA_PARAM
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        print(f"Loading model: {self.model_name}")
        
        if self.model_name == "universal-sentence-encoder":
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
            self.model = hub.load(module_url)
            return
        elif "openelm" in self.model_name.lower():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(device)
            self.tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif self.model_name == "geneol":
            from geneol.geneol_embeddings import get_geneol_embeddings
            from geneol import GenEOL
            args = Namespace(
                model_name_or_path="mistralai/Mistral-7B-v0.1",
                gen_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
                method="s5",
                batch_size=32, 
                num_gens=1,
                max_length=256,
                max_new_tokens=256,
                pooling_method="mean",
                suffix="custom",
                output_folder="./geneol/resultsTF/custom",
                compositional=False,
                task="custom",
                seed=42,
                normalized=True,
                penultimate_layer=-1,
                tsep=False,
                gen_only=False,
                torch_dtype="float16", 
            )
            init_proc = InitProcessGroupKwargs(timeout=1000)
            accelerator = Accelerator(kwargs_handlers=[init_proc])
            self.model = GenEOL(accelerator=accelerator, args=args)
            self.tokenizer = args
        elif "sentence-transformers" in self.model_name:
            self.model = SentenceTransformer(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, device_map="auto")
    
    def _get_embeddings(self, sentences):
        if hasattr(self.model, '_is_hub_module_v1'):
            embeddings = self.model(tf.constant(sentences))
            return torch.from_numpy(embeddings.numpy())
        elif hasattr(self.model, 'encode'):
            all_embeddings = self.model.encode(sentences, args=self.tokenizer)
            return torch.tensor(all_embeddings)
        elif isinstance(self.model, SentenceTransformer):
            return self.model.encode(sentences, convert_to_tensor=True)
        else:
            embeddings = []
            device = next(self.model.parameters()).device
            for sentence in sentences:
                try:
                    encoded = self.tokenizer(sentence, padding=True, truncation=True, 
                                        max_length=512, return_tensors='pt').to(device)
                    with torch.no_grad():
                        # For causal models (like OpenELM), need to enable hidden states output
                        outputs = self.model(**encoded, output_hidden_states=True)
                        
                        # Handle different output types
                        if hasattr(outputs, 'last_hidden_state'):
                            # Standard models (BERT, RoBERTa, etc.)
                            token_embeddings = outputs.last_hidden_state
                        elif hasattr(outputs, 'hidden_states'):
                            # Causal models (GPT, LLaMA, OpenELM, etc.)
                            token_embeddings = outputs.hidden_states[-1]  # Last layer
                        else:
                            print(f"Warning: Unsupported model output type for sentence: {sentence[:50]}...")
                            continue
                        
                        attention_mask = encoded['attention_mask']
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        embeddings.append(embedding.squeeze())
                except Exception as e:
                    print(f"Error processing sentence: {e}")
                    # Add a zero embedding as fallback
                    if embeddings:
                        embeddings.append(torch.zeros_like(embeddings[0]))
                    else:
                        embeddings.append(torch.zeros(768))  # Default embedding size
                
        return torch.stack(embeddings) if embeddings else torch.zeros((len(sentences), 768))
    
    def compute_gain(self, document_sentences, summary_sentences):
        """
        Compute ground truth or model ranking gains
        Returns normalized gains and ranking order.
        """
        doc_embeddings = self._get_embeddings(document_sentences)
        sum_embeddings = self._get_embeddings(summary_sentences)
        
        sims = {}
        for i, doc_emb in enumerate(doc_embeddings):
            similarities = []
            for sum_emb in sum_embeddings:
                sim = cosine_similarity(
                    doc_emb.cpu().numpy().reshape(1, -1),
                    sum_emb.cpu().numpy().reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            sims[i] = np.mean(similarities) if similarities else 0.0

        # Sort by similarity
        sorted_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        
        # Assign relevance N..1 
        N = len(document_sentences)
        gains = {}
        for rank, (idx, _) in enumerate(sorted_items):
            rel = N - rank
            gains[idx] = rel

        # Normalize gains (probabilities)
        norm_factor = sum(gains.values()) if sum(gains.values()) > 0 else 1.0
        for k in gains:
            gains[k] /= norm_factor
        
        ranking = [idx for idx, _ in sorted_items]
        return gains, ranking
    
    def compute_cosine_score(self, reference_text, model_text):
        ref_embedding = self._get_embeddings([reference_text])
        model_embedding = self._get_embeddings([model_text])
        cosine_sim = cosine_similarity(
            ref_embedding.cpu().numpy(),
            model_embedding.cpu().numpy()
        )[0][0]
        return cosine_sim
    
    def compute_ndcg_abs(self, document_text, reference_text, model_text):
        """
        Full nDCG-abs computation
        """
        doc_sentences = tokenize.sent_tokenize(document_text)
        ref_sentences = tokenize.sent_tokenize(reference_text)
        model_sentences = tokenize.sent_tokenize(model_text)
        
        if len(doc_sentences) < 2 or len(ref_sentences) < 1 or len(model_sentences) < 1:
            return {'ndcg_abs': 0.0, 'ndcg': 0.0, 'cosine_score': 0.0, 'dcg': 0.0, 'idcg': 0.0}
        
        # ---- Phase 1: GT gains from reference ----
        gt_gains, gt_ranking = self.compute_gain(doc_sentences, ref_sentences)
        idcg = sum(gt_gains[idx] / np.log2(rank + 2) for rank, idx in enumerate(gt_ranking))
        
        # ---- Phase 2: Model ranking (reuse GT gains) ----
        _, model_ranking = self.compute_gain(doc_sentences, model_sentences)
        dcg = sum(gt_gains.get(idx, 0.0) / np.log2(rank + 2) for rank, idx in enumerate(model_ranking))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # ---- Phase 3: Cosine similarity ----
        cosine_score = self.compute_cosine_score(reference_text, model_text)
        
        # ---- Phase 4: Final score ----
        ndcg_abs_score = self.lambda_param * ndcg + (1 - self.lambda_param) * cosine_score
        
        return {'ndcg_abs': ndcg_abs_score, 'ndcg': ndcg, 'cosine_score': cosine_score, 'dcg': dcg, 'idcg': idcg}

def get_model_path_from_name(model_name):
    for model_path, name in LLM_MODELS:
        if name == model_name:
            return model_path
    return model_name

def evaluate_dataset(data_file="./data/processed_data.json", model_name="sbert-mini", 
                    output_dir="./output"):
    model_path = get_model_path_from_name(model_name)
    
    print(f"nDCG-abs Evaluation")
    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Lambda: {LAMBDA_PARAM}")
    print(f"Summary Type: {SUMMARY_TYPE}")
    print("=" * 50)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    data = [item for item in data if item.get("summary_type") == SUMMARY_TYPE]
    print(f"Evaluating {len(data)} samples...")
    
    evaluator = NDCGAbsEvaluator(model_name=model_path)
    
    results = []
    detailed_results = []
    
    for i, item in enumerate(tqdm(data, desc="Computing nDCG-abs scores")):
        try:
            document = item["Doc"]
            reference = item["Reference"]
            model_summary = item["model"]
            
            result = evaluator.compute_ndcg_abs(document, reference, model_summary)
            
            results.append(result['ndcg_abs'])
            detailed_results.append({
                'sample_id': i,
                'ndcg_abs': result['ndcg_abs'],
                'ndcg': result['ndcg'],
                'cosine_score': result['cosine_score'],
                'dcg': result['dcg'],
                'idcg': result['idcg']
            })
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append(0.0)
            detailed_results.append({
                'sample_id': i,
                'ndcg_abs': 0.0,
                'ndcg': 0.0,
                'cosine_score': 0.0,
                'dcg': 0.0,
                'idcg': 0.0
            })
    
    os.makedirs(output_dir, exist_ok=True)
    scores_csv_file = f"{output_dir}/nDCG_scores.csv"
    
    if os.path.exists(scores_csv_file):
        existing_df = pd.read_csv(scores_csv_file)
        if model_name in existing_df.columns:
            print(f"Warning: Column '{model_name}' already exists. Overwriting...")
            existing_df[model_name] = results
        else:
            existing_df[model_name] = results
        combined_df = existing_df
    else:
        combined_df = pd.DataFrame({model_name: results})
    
    combined_df.to_csv(scores_csv_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Scores CSV: {scores_csv_file}")
    
    print(f"\nStatistics:")
    ndcg_scores = results
    print(f"  Mean nDCG-abs: {np.mean(ndcg_scores):.4f}")
    print(f"  Std nDCG-abs:  {np.std(ndcg_scores):.4f}")
    print(f"  Min nDCG-abs:  {np.min(ndcg_scores):.4f}")
    print(f"  Max nDCG-abs:  {np.max(ndcg_scores):.4f}")
    
    return results, detailed_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ndcg_abs.py <model_name> [model_name2] [model_name3] ...")
        print("Available models:")
        for model_path, model_name in LLM_MODELS:
            print(f"  {model_name} ({model_path})")
        print(f"\nUsing fixed parameters: lambda={LAMBDA_PARAM}, summary_type={SUMMARY_TYPE}")
        print("Examples:")
        print("  python ndcg_abs.py sbert-mini")
        print("  python ndcg_abs.py sbert-mini roberta simcse")
        print("  python ndcg_abs.py all  # Run all available models")
        sys.exit(1)
    
    model_names = sys.argv[1:]
    
    if len(model_names) == 1 and model_names[0].lower() == "all":
        model_names = [name for _, name in LLM_MODELS]
        print(f"Running all {len(model_names)} models: {', '.join(model_names)}")
    
    valid_models = [name for _, name in LLM_MODELS]
    invalid_models = [name for name in model_names if name not in valid_models]
    
    if invalid_models:
        print(f"Error: Invalid model(s): {', '.join(invalid_models)}")
        print("Available models:", ', '.join(valid_models))
        sys.exit(1)
    
    print(f"Starting nDCG-abs evaluation for {len(model_names)} model(s)")
    print("=" * 60)
    
    all_results = {}
    for i, model_name in enumerate(model_names):
        print(f"\n[{i+1}/{len(model_names)}] Processing model: {model_name}")
        print("-" * 50)
        
        try:
            results, detailed_results = evaluate_dataset(model_name=model_name)
            all_results[model_name] = {
                'results': results,
                'detailed_results': detailed_results
            }
            print(f"✓ Completed {model_name}")
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print(f"EVALUATION COMPLETE")
    print(f"Successfully processed: {len(all_results)}/{len(model_names)} models")
    
    if all_results:
        print(f"\nModel Performance Summary:")
        print("-" * 40)
        for model_name, data in all_results.items():
            scores = data['results']
            print(f"{model_name:<15}: μ={np.mean(scores):.4f}, σ={np.std(scores):.4f}")
        
        print(f"\nResults saved to: ./output/nDCG_scores.csv")
        print(f"Lambda parameter: {LAMBDA_PARAM}")
    else:
        print("No models were successfully processed.")
