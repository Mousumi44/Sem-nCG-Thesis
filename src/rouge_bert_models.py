import json
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
import evaluate
from tqdm import tqdm

def load_data(file_path="./data/processed_data.json", summary_type="Abstractive"):
    """Load and filter data by summary type"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter by summary type
    filtered_data = [item for item in data if item.get("summary_type") == summary_type]
    print(f"Loaded {len(filtered_data)} {summary_type} samples")
    
    return filtered_data

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
    return {
        'bertscore': results['f1']  # Use only F1, which is the main BERTScore
    }

def extract_human_scores(data):
    """Extract human annotation scores"""
    human_scores = {"coherence": [], "consistency": [], "fluency": [], "relevance": []}
    
    for entry in data:
        if "expert_annotations" in entry and entry["expert_annotations"]:
            anns = entry["expert_annotations"]
            for key in human_scores:
                vals = [ann.get(key, np.nan) for ann in anns]
                avg = np.nanmean(vals) if vals else np.nan
                human_scores[key].append(avg)
        else:
            for key in human_scores:
                human_scores[key].append(np.nan)
    
    return human_scores

def compute_kendall_tau(metric_scores, human_scores):
    """Compute Kendall's tau correlation between metric and human scores"""
    results = []
    
    for metric_name, scores in metric_scores.items():
        print(f"\nMetric: {metric_name}")
        
        for annotation_type, human_vals in human_scores.items():
            # Remove pairs where human score is nan
            valid_pairs = [(h, s) for h, s in zip(human_vals, scores) 
                          if not np.isnan(h) and not np.isnan(s)]
            
            if len(valid_pairs) < 2:
                print(f"  {annotation_type}: Not enough data ({len(valid_pairs)} valid pairs)")
                results.append({
                    "metric": metric_name, 
                    "annotation": annotation_type, 
                    "tau": None, 
                    "p": None,
                    "valid_pairs": len(valid_pairs)
                })
                continue
            
            h_vals, s_vals = zip(*valid_pairs)
            tau, p = kendalltau(h_vals, s_vals)
            print(f"  {annotation_type}: tau={tau:.3f}, p={p:.3g} ({len(valid_pairs)} pairs)")
            
            results.append({
                "metric": metric_name,
                "annotation": annotation_type,
                "tau": tau,
                "p": p,
                "valid_pairs": len(valid_pairs)
            })
    
    return results

def save_metric_scores(rouge_scores, bert_scores, output_dir="./output"):
    """Save individual metric scores to files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all scores
    all_scores = {**rouge_scores, **bert_scores}
    
    # Save to CSV with samples as rows and metrics as columns
    df_scores = pd.DataFrame(all_scores)
    scores_file = f"{output_dir}/rouge_bert_scores.csv"
    df_scores.to_csv(scores_file, index=False)
    print(f"Metric scores saved to {scores_file}")
    
    return all_scores

def main():
    print("ROUGE and BERTScore Evaluation")
    print("=" * 50)
    
    # Load data
    data = load_data(summary_type="Abstractive")
    
    # Extract references and predictions
    references = [item["Reference"] for item in data]
    predictions = [item["model"] for item in data]  # Generated summaries
    
    print(f"Evaluating {len(references)} summary pairs...")
    
    # Compute ROUGE scores
    print("\nComputing ROUGE scores...")
    rouge_scores = compute_rouge_scores(references, predictions)
    
    # Compute BERTScore
    print("Computing BERTScore...")
    bert_scores = compute_bertscore(references, predictions)
    
    # Extract human scores
    print("Extracting human annotation scores...")
    human_scores = extract_human_scores(data)
    
    # Save metric scores
    all_metric_scores = save_metric_scores(rouge_scores, bert_scores)
    
    # Compute Kendall's tau correlations
    print("\nComputing Kendall's tau correlations...")
    kendall_results = compute_kendall_tau(all_metric_scores, human_scores)
    
    # Save Kendall's tau results
    df_kendall = pd.DataFrame(kendall_results)
    kendall_file = "./output/rouge_bert_kendall_results.csv"
    df_kendall.to_csv(kendall_file, index=False)
    print(f"\nKendall's tau results saved to {kendall_file}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    for metric_name, scores in all_metric_scores.items():
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            print(f"{metric_name:15s}: mean={np.mean(valid_scores):.3f}, "
                  f"std={np.std(valid_scores):.3f}, "
                  f"min={np.min(valid_scores):.3f}, "
                  f"max={np.max(valid_scores):.3f}")
    
    # Print best correlations
    print(f"\n{'BEST CORRELATIONS'}")
    print("-" * 30)
    df_valid = df_kendall[df_kendall['tau'].notna()]
    if not df_valid.empty:
        for annotation in ['coherence', 'consistency', 'fluency', 'relevance']:
            ann_data = df_valid[df_valid['annotation'] == annotation]
            if not ann_data.empty:
                best_row = ann_data.loc[ann_data['tau'].abs().idxmax()]
                print(f"{annotation:12s}: {best_row['metric']:15s} "
                      f"tau={best_row['tau']:.3f} (p={best_row['p']:.3g})")

if __name__ == "__main__":
    main()
