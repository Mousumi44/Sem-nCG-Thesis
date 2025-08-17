#!/usr/bin/env python3
"""
Evaluate nDCG-abs correlations with human judgments on MTurk XSum dataset
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.append('./src')
from ndcg_abs import NDCGAbsEvaluator

# Model mapping
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

def get_model_path_from_name(model_name):
    """Get the model path from the model name using LLM_MODELS list"""
    for model_path, name in LLM_MODELS:
        if name == model_name:
            return model_path
    return model_name  # If not found, assume it's already a path

def load_mturk_xsum_data(filepath):
    """Load MTurk XSum dataset"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def compute_human_scores(responses):
    """
    Convert human annotations to scores
    
    Args:
        responses: List of worker responses [{"worker_id": X, "response": "yes/no"}]
    
    Returns:
        Human score (proportion of "yes" responses)
    """
    if not responses:
        return 0.0
    
    yes_count = sum(1 for r in responses if r.get("response", "").lower() == "yes")
    return yes_count / len(responses)

def evaluate_correlations(data_file="./data/mturk_xsum.jsonl", model_name="sbert-mini", max_samples=None):
    """
    Evaluate nDCG-abs correlations with human judgments
    
    Args:
        data_file: Path to MTurk XSum dataset
        model_name: Model to use for nDCG computation
        max_samples: Maximum number of samples to process (None for all)
    """
    # Convert model name to model path if needed
    model_path = get_model_path_from_name(model_name)
    
    print("Loading MTurk XSum dataset...")
    data = load_mturk_xsum_data(data_file)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {max_samples} samples...")
    else:
        print(f"Processing all {len(data)} samples...")
    
    print(f"Loading model: {model_name}")
    print(f"Model path: {model_path}")
    
    # Initialize nDCG evaluator with the model path
    evaluator = NDCGAbsEvaluator(model_name=model_path)
    
    # Store results
    ndcg_scores = []
    human_scores = []
    
    print("Computing nDCG-abs scores and human judgments...")
    for i, item in enumerate(tqdm(data, desc="Processing samples")):
        try:
            article = item["article"]
            
            # Process each summary sentence
            for sent_data in item["summary_sentences"]:
                sentence = sent_data["sentence"]
                responses = sent_data["responses"]
                
                # Compute human score (proportion of "yes" responses)
                human_score = compute_human_scores(responses)
                
                # Compute nDCG-abs score for factual support
                # For factual support, we evaluate how well the sentence is supported by the article
                # We use the sentence as both reference and model (since we're measuring factual consistency)
                # The key insight: nDCG measures how well the model summary (sentence) 
                # captures information from the document (article) relative to a reference
                # For factual support, the sentence should capture factual content from the article
                result = evaluator.compute_ndcg_abs(article, sentence, sentence)
                ndcg_score = result['ndcg_abs']
                
                # Store results
                ndcg_scores.append(ndcg_score)
                human_scores.append(human_score)
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Compute correlations
    print(f"\nComputed scores for {len(ndcg_scores)} sentences")
    
    if len(ndcg_scores) < 2:
        print("Error: Not enough valid samples for correlation analysis")
        return
    
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(ndcg_scores, human_scores)
    
    # Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(ndcg_scores, human_scores)
    
    # Kendall's tau
    kendall_corr, kendall_p = kendalltau(ndcg_scores, human_scores)
    
    # Print results
    print(f"\nModel: {model_name}")
    print(f"Samples: {len(ndcg_scores)} | Kendall τ: {kendall_corr:.4f} (p={kendall_p:.4f})")
    
    # Save results
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save/append to correlation CSV summary
    correlation_csv_file = f"{output_dir}/correlation_xsum.csv"
    
    # Create new row for this model
    new_row = pd.DataFrame({
        'model': [model_name],
        'kendall_tau': [float(kendall_corr)],
        'kendall_p': [float(kendall_p)],
        'n_samples': [len(ndcg_scores)],
        'pearson_r': [float(pearson_corr)],
        'spearman_rho': [float(spearman_corr)]
    })
    
    # Check if CSV exists and append, otherwise create new
    if os.path.exists(correlation_csv_file):
        # Read existing data
        existing_df = pd.read_csv(correlation_csv_file)
        # Check if this model already exists
        if model_name in existing_df['model'].values:
            existing_df = existing_df[existing_df['model'] != model_name]
        # Add new row
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        # Create new file
        combined_df = new_row
    
    # Sort by kendall_tau descending
    combined_df = combined_df.sort_values('kendall_tau', ascending=False)
    combined_df.to_csv(correlation_csv_file, index=False)
    
    print(f"Results saved to: {correlation_csv_file}")
    
    return {
        'model': model_name,
        'kendall_tau': float(kendall_corr),
        'kendall_p': float(kendall_p),
        'n_samples': len(ndcg_scores)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate nDCG-abs correlations with MTurk XSum human judgments")
    parser.add_argument("--model", default="sbert-mini", help="Model name to use")
    parser.add_argument("--data", default="./data/mturk_xsum.jsonl", help="Path to MTurk XSum dataset")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    
    args = parser.parse_args()
    
    evaluate_correlations(
        data_file=args.data,
        model_name=args.model,
        max_samples=args.max_samples
    )
