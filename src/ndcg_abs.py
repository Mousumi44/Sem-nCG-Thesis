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

# Use the same LLM_MODELS list from main.py
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
LAMBDA_PARAM = 0.8 
SUMMARY_TYPE = "Abstractive"  # Fixed to Abstractive summaries only

class NDCGAbsEvaluator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the nDCG-abs evaluator
        
        Args:
            model_name: Model to use for sentence embeddings
        """
        self.lambda_param = LAMBDA_PARAM
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model using the same logic as main.py"""
        print(f"Loading model: {self.model_name}")
        
        if self.model_name == "universal-sentence-encoder":
            # For Universal Sentence Encoder, load from TF Hub
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
            self.model = hub.load(module_url)
            return
        elif "openelm" in self.model_name.lower():
            # Use LlamaTokenizer as a compatible tokenizer for OpenELM
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(device)
            self.tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif self.model_name == "geneol":
            # For GenEOL model
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
            # For sentence transformer models
            self.model = SentenceTransformer(self.model_name)
        else:
            # For other transformer models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, device_map="auto")
    
    def _get_embeddings(self, sentences):
        """Get embeddings for a list of sentences using the same logic as main.py"""
        if hasattr(self.model, '_is_hub_module_v1'):
            # For Universal Sentence Encoder
            embeddings = self.model(tf.constant(sentences))
            return torch.from_numpy(embeddings.numpy())
        elif hasattr(self.model, 'encode'):
            # For GenEOL and similar models with encode method
            all_embeddings = self.model.encode(sentences, args=self.tokenizer)
            return torch.tensor(all_embeddings)
        elif isinstance(self.model, SentenceTransformer):
            # For sentence transformer models
            return self.model.encode(sentences, convert_to_tensor=True)
        else:
            # For other transformer models - use mean pooling like in main.py
            embeddings = []
            device = next(self.model.parameters()).device
            for sentence in sentences:
                encoded = self.tokenizer(sentence, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # Mean pooling
                    attention_mask = encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings.append(embedding.squeeze())
            return torch.stack(embeddings)
    
    def compute_gain(self, document_sentences, reference_sentences):
        """
        
        Args:
            document_sentences: List of sentences from the document
            reference_sentences: List of sentences from the reference summary
            
        Returns:
            GTgain: Dictionary mapping sentence indices to normalized gains
        """
        # Step 4: Represent sentences by embedding vectors
        doc_embeddings = self._get_embeddings(document_sentences)
        ref_embeddings = self._get_embeddings(reference_sentences)
        
        GT = {}
        
        # Compute simple cosine similarities between document and reference sentences
        for i, doc_emb in enumerate(doc_embeddings):
            similarities = []
            for ref_emb in ref_embeddings:
                sim = cosine_similarity(
                    doc_emb.cpu().numpy().reshape(1, -1),
                    ref_emb.cpu().numpy().reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            # Use max similarity to reference sentences for each document sentence
            GT[i] = np.max(similarities) if similarities else 0.0
        
        # Step 11: GTsorted ← Sort GT based on mean(Sim)
        GT_sorted = sorted(GT.items(), key=lambda x: x[1], reverse=True)
        
        # Create rank mapping
        rank_map = {sent_idx: rank + 1 for rank, (sent_idx, _) in enumerate(GT_sorted)}
        
        # Steps 12-15: Compute GTgain
        GTgain = {}
        N = len(document_sentences)
        
        for i, doc_sent in enumerate(document_sentences):
            # Step 13: SiD(rel) ← N − rank(SiD,GTsorted) + 1
            rel_score = N - rank_map[i] + 1
            
            # Step 14: GTgain[SiD] ← SiD(rel) / log2(i+1)
            # Note: Using position in sorted order for DCG calculation
            position = rank_map[i]
            GTgain[i] = rel_score / np.log2(position + 1)
        
        # Step 16: Normalize GTgain into a probabilistic gain
        gain_values = list(GTgain.values())
        if sum(gain_values) > 0:
            total_gain = sum(gain_values)
            GTgain = {k: v / total_gain for k, v in GTgain.items()}
        
        return GTgain
    
    def compute_dcg(self, document_sentences, summary_sentences, gt_gain=None):
        """
        Compute DCG score using the groundtruth gain
        
        Args:
            document_sentences: List of document sentences
            summary_sentences: List of summary sentences  
            gt_gain: Precomputed groundtruth gain (if None, will compute)
            
        Returns:
            DCG score
        """
        if gt_gain is None:
            # If no precomputed gain, compute it (this shouldn't happen in normal flow)
            gt_gain = self.compute_gain(document_sentences, summary_sentences)
        
        # Get embeddings
        doc_embeddings = self._get_embeddings(document_sentences)
        sum_embeddings = self._get_embeddings(summary_sentences)
        
        # Find most relevant document sentences for each summary sentence
        dcg_score = 0.0
        
        for i, sum_sent in enumerate(summary_sentences):
            # Find the document sentence most similar to this summary sentence
            max_sim = -1
            best_doc_idx = 0
            
            for j, doc_sent in enumerate(document_sentences):
                # Use simple cosine similarity
                sim = cosine_similarity(
                    sum_embeddings[i].cpu().numpy().reshape(1, -1),
                    doc_embeddings[j].cpu().numpy().reshape(1, -1)
                )[0][0]
                
                if sim > max_sim:
                    max_sim = sim
                    best_doc_idx = j
            
            # Add the gain from the most relevant document sentence
            # Discounted by position in summary
            gain = gt_gain.get(best_doc_idx, 0.0)
            dcg_score += gain / np.log2(i + 2)  # i+2 because DCG starts from position 1
        
        return dcg_score
    
    def compute_idcg(self, gt_gain, k=None):
        """
        Compute Ideal DCG using the groundtruth gain
        
        Args:
            gt_gain: Groundtruth gain dictionary
            k: Number of top positions to consider (if None, use all)
            
        Returns:
            IDCG score
        """
        # Sort gains in descending order
        sorted_gains = sorted(gt_gain.values(), reverse=True)
        
        if k is not None:
            sorted_gains = sorted_gains[:k]
        
        idcg_score = 0.0
        for i, gain in enumerate(sorted_gains):
            idcg_score += gain / np.log2(i + 2)  # i+2 because DCG starts from position 1
        
        return idcg_score
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text for better evaluation
        """
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
        
        # Remove very short sentences (less than 10 characters)
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return '. '.join(sentences) + '.' if sentences else text
    
    def filter_sentences(self, sentences):
        """
        Filter out low-quality sentences
        """
        import re
        filtered = []
        for sent in sentences:
            sent = sent.strip()
            # Skip very short sentences
            if len(sent) < 15:
                continue
            # Skip sentences with too many special characters
            if len(re.findall(r'[^a-zA-Z0-9\s]', sent)) > len(sent) * 0.3:
                continue
            # Skip sentences that are mostly numbers
            if len(re.findall(r'\d', sent)) > len(sent) * 0.5:
                continue
            filtered.append(sent)
        return filtered if filtered else sentences  # Return original if all filtered out
    

    def compute_cosine_score(self, reference_text, model_text):
        """
        Phase 3: Cosine Similarity Score between reference and model summary
        
        Args:
            reference_text: Reference summary text
            model_text: Model-generated summary text
            
        Returns:
            Cosine similarity score
        """
        ref_embedding = self._get_embeddings([reference_text])
        model_embedding = self._get_embeddings([model_text])
        
        cosine_sim = cosine_similarity(
            ref_embedding.cpu().numpy(),
            model_embedding.cpu().numpy()
        )[0][0]
        
        return cosine_sim
    
    def compute_ndcg_abs(self, document_text, reference_text, model_text):
        """
        Complete nDCG-abs computation following Algorithm 1 (with preprocessing)
        
        Args:
            document_text: Original document text
            reference_text: Reference summary text
            model_text: Model-generated summary text
            
        Returns:
            nDCG-abs score
        """
        # Preprocess texts
        document_text = self.preprocess_text(document_text)
        reference_text = self.preprocess_text(reference_text)
        model_text = self.preprocess_text(model_text)
        
        # Tokenize into sentences and filter
        doc_sentences = self.filter_sentences(tokenize.sent_tokenize(document_text))
        ref_sentences = self.filter_sentences(tokenize.sent_tokenize(reference_text))
        model_sentences = self.filter_sentences(tokenize.sent_tokenize(model_text))
        
        # Skip very short texts that might cause issues
        if len(doc_sentences) < 2 or len(ref_sentences) < 1 or len(model_sentences) < 1:
            return {
                'ndcg_abs': 0.0,
                'ndcg': 0.0,
                'cosine_score': 0.0,
                'dcg': 0.0,
                'idcg': 0.0
            }
        
        # Phase 1: Groundtruth Gain Computation
        gt_gain = self.compute_gain(doc_sentences, ref_sentences)
        
        # Phase 2: nDCG Computation
        # Step 1: IDCG ← Compute the IDCG score using COMPUTEGAIN(D,R)
        idcg = self.compute_idcg(gt_gain)
        
        # Step 2: DCG ← Compute the DCG score using COMPUTEGAIN(D,M)
        dcg = self.compute_dcg(doc_sentences, model_sentences, gt_gain)
        
        # Step 3: nDCG ← DCG/IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Phase 3: Cosine Similarity Score
        cosine_score = self.compute_cosine_score(reference_text, model_text)
        
        # Phase 4: Final Score
        # nDCG-abs Score = λ · nDCG + (1 − λ) · Cosine Score
        ndcg_abs_score = self.lambda_param * ndcg + (1 - self.lambda_param) * cosine_score
        
        return {
            'ndcg_abs': ndcg_abs_score,
            'ndcg': ndcg,
            'cosine_score': cosine_score,
            'dcg': dcg,
            'idcg': idcg
        }

def get_model_path_from_name(model_name):
    """Get the model path from the model name using LLM_MODELS list"""
    for model_path, name in LLM_MODELS:
        if name == model_name:
            return model_path
    return model_name  # If not found, assume it's already a path

def evaluate_dataset(data_file="./data/processed_data.json", model_name="sbert-mini", 
                    output_dir="./output"):
    """
    Evaluate entire dataset using nDCG-abs
    
    Args:
        data_file: Path to the dataset file
        model_name: Model name from LLM_MODELS list or direct model path
        output_dir: Output directory for results
    """
    # Convert model name to model path if needed
    model_path = get_model_path_from_name(model_name)
    
    print(f"nDCG-abs Evaluation")
    print(f"Model Name: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Lambda: {LAMBDA_PARAM}")
    print(f"Summary Type: {SUMMARY_TYPE}")
    print("=" * 50)
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Filter by summary type (always Abstractive)
    data = [item for item in data if item.get("summary_type") == SUMMARY_TYPE]
    
    print(f"Evaluating {len(data)} samples...")
    
    # Initialize evaluator with the model path
    evaluator = NDCGAbsEvaluator(model_name=model_path)
    
    results = []
    detailed_results = []
    
    for i, item in enumerate(tqdm(data, desc="Computing nDCG-abs scores")):
        try:
            document = item["Doc"]
            reference = item["Reference"]
            model_summary = item["model"]
            
            # Compute nDCG-abs
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
    
    # Save results in the same format as rouge_bert_scores.csv
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a single shared file for all models (like rouge_bert_scores.csv)
    scores_csv_file = f"{output_dir}/nDCG_scores.csv"
    
    # Create DataFrame with just the scores (no headers, no metadata)
    new_scores_df = pd.DataFrame({model_name: results})
    
    # Check if file exists and append as new column, otherwise create new file
    if os.path.exists(scores_csv_file):
        # Read existing data
        existing_df = pd.read_csv(scores_csv_file)
        # Check if this model column already exists
        if model_name in existing_df.columns:
            print(f"Warning: Column '{model_name}' already exists. Overwriting...")
            existing_df = existing_df.drop(columns=[model_name])
        # Add new column
        combined_df = pd.concat([existing_df, new_scores_df], axis=1)
    else:
        # Create new file
        combined_df = new_scores_df
    
    # Save updated CSV
    combined_df.to_csv(scores_csv_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Scores CSV: {scores_csv_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Mean nDCG-abs: {np.mean(results):.4f}")
    print(f"  Std nDCG-abs:  {np.std(results):.4f}")
    print(f"  Min nDCG-abs:  {np.min(results):.4f}")
    print(f"  Max nDCG-abs:  {np.max(results):.4f}")
    
    return results, detailed_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ndcg_abs.py <model_name>")
        print("Available models:")
        for model_path, model_name in LLM_MODELS:
            print(f"  {model_name} ({model_path})")
        print(f"\nUsing fixed parameters: lambda={LAMBDA_PARAM}, summary_type={SUMMARY_TYPE}")
        print("Example: python ndcg_abs.py sbert-mini")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Validate model name
    valid_models = [name for _, name in LLM_MODELS]
    if model_name not in valid_models:
        print(f"Error: Model '{model_name}' not found in LLM_MODELS.")
        print("Available models:", ', '.join(valid_models))
        sys.exit(1)
    
    evaluate_dataset(model_name=model_name)
