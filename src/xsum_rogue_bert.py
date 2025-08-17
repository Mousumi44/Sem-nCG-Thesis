#!/usr/bin/env python3
"""
Evaluate ROUGE and BERTScore correlations with human judgments on MTurk XSum dataset
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm
import sys
import os
import evaluate

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

def compute_rouge_scores(references, predictions):
    """Compute ROUGE scores using HuggingFace evaluate"""
    rouge = evaluate.load("rouge")
    
    # Compute ROUGE scores
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=False  # Get scores for each sample
    )
    
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'], 
        'rougeL': results['rougeL'],
        'rougeLsum': results['rougeLsum']
    }

def compute_bertscore(references, predictions):
    """Compute BERTScore using HuggingFace evaluate"""
    bertscore = evaluate.load("bertscore")
    
    # Compute BERTScore with default model (roberta-large)
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    
    # Only return F1 score (the standard BERTScore metric)
    return results['f1']

def evaluate_rouge_bert_correlations(data_file="./data/mturk_xsum.jsonl", max_samples=None):
    """
    Evaluate ROUGE and BERTScore correlations with human judgments
    
    Args:
        data_file: Path to MTurk XSum dataset
        max_samples: Maximum number of samples to process (None for all)
    """
    print("Loading MTurk XSum dataset...")
    data = load_mturk_xsum_data(data_file)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Processing first {max_samples} samples...")
    else:
        print(f"Processing all {len(data)} samples...")
    
    # Store results
    references = []
    predictions = []
    human_scores = []
    
    print("Preparing data for ROUGE and BERTScore evaluation...")
    for i, item in enumerate(tqdm(data, desc="Processing samples")):
        try:
            article = item["article"]
            
            # Process each summary sentence
            for sent_data in item["summary_sentences"]:
                sentence = sent_data["sentence"]
                responses = sent_data["responses"]
                
                # Compute human score (proportion of "yes" responses)
                human_score = compute_human_scores(responses)
                
                # For ROUGE/BERTScore, we compare sentence to article (as reference)
                references.append(article)
                predictions.append(sentence)
                human_scores.append(human_score)
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"\nPrepared {len(references)} sentence pairs for evaluation")
    
    if len(references) < 2:
        print("Error: Not enough valid samples for evaluation")
        return
    
    # Compute ROUGE scores
    print("Computing ROUGE scores...")
    rouge_results = compute_rouge_scores(references, predictions)
    
    # Compute BERTScore
    print("Computing BERTScore...")
    bert_scores = compute_bertscore(references, predictions)
    
    # Combine all metric scores
    all_metrics = {
        'rouge1': rouge_results['rouge1'],
        'rouge2': rouge_results['rouge2'],
        'rougeL': rouge_results['rougeL'],
        'rougeLsum': rouge_results['rougeLsum'],
        'bertscore': bert_scores
    }
    
    # Compute correlations for each metric
    correlation_results = []
    
    print("\nComputing correlations with human judgments...")
    for metric_name, metric_scores in all_metrics.items():
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(metric_scores, human_scores)
        spearman_corr, spearman_p = spearmanr(metric_scores, human_scores)
        kendall_corr, kendall_p = kendalltau(metric_scores, human_scores)
        
        correlation_results.append({
            'model': metric_name,
            'kendall_tau': float(kendall_corr),
            'kendall_p': float(kendall_p),
            'n_samples': len(metric_scores),
            'pearson_r': float(pearson_corr),
            'spearman_rho': float(spearman_corr)
        })
        
        print(f"{metric_name:12s}: Kendall τ = {kendall_corr:.4f} (p={kendall_p:.4f})")
    
    # Save/append to correlation CSV summary
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    correlation_csv_file = f"{output_dir}/correlation_xsum.csv"
    
    # Create DataFrame for new results
    new_results_df = pd.DataFrame(correlation_results)
    
    # Check if CSV exists and append, otherwise create new
    if os.path.exists(correlation_csv_file):
        # Read existing data
        existing_df = pd.read_csv(correlation_csv_file)
        
        # Remove any existing entries for these metrics
        metric_names = [r['model'] for r in correlation_results]
        existing_df = existing_df[~existing_df['model'].isin(metric_names)]
        
        # Add new rows
        combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
    else:
        # Create new file
        combined_df = new_results_df
    
    # Sort by kendall_tau descending
    combined_df = combined_df.sort_values('kendall_tau', ascending=False)
    combined_df.to_csv(correlation_csv_file, index=False)
    
    # Save detailed results
    detailed_results = []
    for i, (ref, pred, human_score) in enumerate(zip(references, predictions, human_scores)):
        result_dict = {
            'sample_id': i,
            'sentence': pred,
            'human_score': human_score
        }
        # Add all metric scores
        for metric_name, metric_scores in all_metrics.items():
            result_dict[metric_name] = metric_scores[i]
        detailed_results.append(result_dict)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = f"{output_dir}/mturk_xsum_rouge_bert_results.csv"
    detailed_df.to_csv(detailed_file, index=False)
    
    # Print results
    print(f"\nROUGE & BERTScore results:")
    for result in correlation_results:
        print(f"{result['model']:12s}: Kendall τ = {result['kendall_tau']:6.4f} (p={result['kendall_p']:.4f})")
    
    print(f"\nResults saved to: {correlation_csv_file}")
    
    return correlation_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ROUGE/BERTScore correlations with MTurk XSum human judgments")
    parser.add_argument("--data", default="./data/mturk_xsum.jsonl", help="Path to MTurk XSum dataset")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    
    args = parser.parse_args()
    
    evaluate_rouge_bert_correlations(
        data_file=args.data,
        max_samples=args.max_samples
    )
