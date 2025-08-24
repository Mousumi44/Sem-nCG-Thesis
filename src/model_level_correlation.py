#!/usr/bin/env python3
"""
Model-Level Kendall's Tau Correlation Analysis
Computes correlations between automatic metrics and human annotations 
by averaging across summaries per model (12 models).
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import os

def load_data():
    """Load the processed dataset"""
    with open("./data/processed_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter to abstractive summaries only
    abstractive_data = [item for item in data if item.get("summary_type") == "Abstractive"]
    print(f"Loaded {len(abstractive_data)} abstractive summaries")
    
    return abstractive_data

def compute_model_level_human_scores(data):
    """
    Compute average human scores per model.
    For each model, average across all summaries and all annotators.
    """
    model_human_scores = {}
    
    for item in data:
        model_id = item.get('model_id', 'unknown')
        
        if model_id not in model_human_scores:
            model_human_scores[model_id] = {
                'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []
            }
        
        # Get human annotations for this summary
        if "expert_annotations" in item and item["expert_annotations"]:
            annotations = item["expert_annotations"]
            
            # For each aspect, average across annotators and add to model
            for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
                aspect_scores = []
                for ann in annotations:
                    if aspect in ann and ann[aspect] is not None:
                        try:
                            aspect_scores.append(float(ann[aspect]))
                        except (ValueError, TypeError):
                            continue
                
                if aspect_scores:
                    model_human_scores[model_id][aspect].append(np.mean(aspect_scores))
                else:
                    model_human_scores[model_id][aspect].append(np.nan)
        else:
            # No annotations for this summary
            for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
                model_human_scores[model_id][aspect].append(np.nan)
    
    # Average across all summaries for each model
    model_averages = {}
    for model_id, scores_dict in model_human_scores.items():
        model_averages[model_id] = {}
        for aspect, scores in scores_dict.items():
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                model_averages[model_id][aspect] = np.mean(valid_scores)
            else:
                model_averages[model_id][aspect] = np.nan
    
    return model_averages

def load_automatic_metrics():
    """Load automatic metric scores from baseline models and nDCG"""
    metrics_data = {}
    
    # # Load baseline metrics
    # try:
    #     df_baseline = pd.read_csv("./output/baseline_models_scores.csv")
    #     print(f"Loaded baseline metrics: {df_baseline.shape[0]} rows, {len(df_baseline.columns)} metrics")
    #     print(f"Baseline metrics: {list(df_baseline.columns)}")
    #     metrics_data['baseline'] = df_baseline
    # except FileNotFoundError:
    #     print("Warning: ./output/baseline_models_scores.csv not found")
    #     print("Please run src/baseline_models.py first to generate the baseline scores")
    #     metrics_data['baseline'] = None
    
    # Load nDCG metrics
    try:
        df_ndcg = pd.read_csv("./output/nDCG_scores.csv")
        print(f"Loaded nDCG metrics: {df_ndcg.shape[0]} rows, {len(df_ndcg.columns)} metrics")
        print(f"nDCG metrics: {list(df_ndcg.columns)}")
        metrics_data['ndcg'] = df_ndcg
    except FileNotFoundError:
        print("Warning: ./output/nDCG_scores.csv not found")
        print("Please run src/ndcg_abs.py to generate nDCG scores")
        metrics_data['ndcg'] = None
    
    return metrics_data

def compute_model_level_metric_scores(metrics_data, data):
    """
    Compute average metric scores per model using the generated scores.
    Handles both baseline metrics (with model_id) and nDCG metrics (positional).
    """
    all_model_averages = {}
    
    # # Process baseline metrics (with model_id column)
    # if metrics_data['baseline'] is not None:
    #     df_baseline = metrics_data['baseline']
        
    #     # Check if CSV has model_id column
    #     if 'model_id' in df_baseline.columns:
    #         print("Processing baseline metrics using model_id column")
            
    #         # Group by model_id directly from CSV
    #         for _, row in df_baseline.iterrows():
    #             model_id = row['model_id']
                
    #             if model_id not in all_model_averages:
    #                 all_model_averages[model_id] = {}
                
    #             # Add baseline metric scores to the model
    #             for col in df_baseline.columns:
    #                 if col == 'model_id':
    #                     continue
    #                 try:
    #                     score = float(row[col])
    #                     if model_id not in all_model_averages:
    #                         all_model_averages[model_id] = {}
    #                     if col not in all_model_averages[model_id]:
    #                         all_model_averages[model_id][col] = []
    #                     all_model_averages[model_id][col].append(score)
    #                 except (ValueError, TypeError):
    #                     continue
    #     else:
    #         print("WARNING: No model_id column found in baseline CSV! Using positional alignment")
    #         # Positional alignment: CSV row i corresponds to data sample i
    #         for i, item in enumerate(data):
    #             if i >= len(df_baseline):
    #                 break
                    
    #             model_id = item.get('model_id', 'unknown')
                
    #             if model_id not in all_model_averages:
    #                 all_model_averages[model_id] = {}
                
    #             # CSV row i corresponds to data sample i with model_id from data
    #             for col in df_baseline.columns:
    #                 try:
    #                     score = float(df_baseline.iloc[i][col])
    #                     if col not in all_model_averages[model_id]:
    #                         all_model_averages[model_id][col] = []
    #                     all_model_averages[model_id][col].append(score)
    #                 except (ValueError, TypeError, IndexError):
    #                     continue
    
    # Process nDCG metrics (now with model_id column like baseline metrics)
    if metrics_data['ndcg'] is not None:
        df_ndcg = metrics_data['ndcg']
        
        # Check if nDCG CSV has model_id column
        if 'model_id' in df_ndcg.columns:
            print("Processing nDCG metrics using model_id column")
            
            # Group by model_id directly from CSV (same as baseline)
            for _, row in df_ndcg.iterrows():
                model_id = row['model_id']
                
                if model_id not in all_model_averages:
                    all_model_averages[model_id] = {}
                
                # Add nDCG metric scores to the model
                for col in df_ndcg.columns:
                    if col == 'model_id':
                        continue
                    ndcg_col_name = f"ndcg_{col}"  # Prefix to distinguish from baseline metrics
                    try:
                        score = float(row[col])
                        if ndcg_col_name not in all_model_averages[model_id]:
                            all_model_averages[model_id][ndcg_col_name] = []
                        all_model_averages[model_id][ndcg_col_name].append(score)
                    except (ValueError, TypeError):
                        continue
        else:
            print("WARNING: No model_id column found in nDCG CSV! Using positional alignment")
            # Positional alignment: CSV row i corresponds to data sample i
            for i, item in enumerate(data):
                if i >= len(df_ndcg):
                    print(f"Warning: nDCG CSV has fewer rows ({len(df_ndcg)}) than data samples ({len(data)})")
                    break
                    
                model_id = item.get('model_id', 'unknown')
                
                if model_id not in all_model_averages:
                    all_model_averages[model_id] = {}
                
                # Add nDCG scores for this sample using positional mapping
                # CSV row i corresponds to data sample i with model_id from data
                for col in df_ndcg.columns:
                    ndcg_col_name = f"ndcg_{col}"  # Prefix to distinguish from baseline metrics
                    try:
                        score = float(df_ndcg.iloc[i][col])
                        if ndcg_col_name not in all_model_averages[model_id]:
                            all_model_averages[model_id][ndcg_col_name] = []
                        all_model_averages[model_id][ndcg_col_name].append(score)
                    except (ValueError, TypeError, IndexError):
                        continue
        
        print(f"Successfully processed nDCG metrics for {len(all_model_averages)} models")
    
    # Average across all summaries for each model
    model_averages = {}
    for model_id, scores_dict in all_model_averages.items():
        model_averages[model_id] = {}
        for metric, scores in scores_dict.items():
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                model_averages[model_id][metric] = np.mean(valid_scores)
            else:
                model_averages[model_id][metric] = np.nan
    
    return model_averages

def compute_model_level_correlations(model_human_scores, model_metric_scores):
    """
    Compute Kendall's tau correlations at model level.
    Each correlation uses 12 data points (one per model).
    """
    results = []
    
    # Get common model IDs
    common_models = set(model_human_scores.keys()) & set(model_metric_scores.keys())
    common_models = sorted(list(common_models))
    
    print(f"\nComputing correlations for {len(common_models)} models: {common_models}")
    
    if len(common_models) < 3:
        print("Error: Need at least 3 models for correlation analysis")
        return []
    
    # For each metric
    for metric_name in model_metric_scores[common_models[0]].keys():
        print(f"\nProcessing metric: {metric_name}")
        
        # For each human aspect
        for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
            # Build arrays for this metric-aspect pair
            metric_values = []
            human_values = []
            
            for model_id in common_models:
                metric_val = model_metric_scores[model_id][metric_name]
                human_val = model_human_scores[model_id][aspect]
                
                if not np.isnan(metric_val) and not np.isnan(human_val):
                    metric_values.append(metric_val)
                    human_values.append(human_val)
            
            print(f"  {aspect}: {len(metric_values)} valid model pairs")
            
            if len(metric_values) < 3:
                print(f"    Not enough data for correlation ({len(metric_values)} pairs)")
                results.append({
                    'metric': metric_name,
                    'aspect': aspect,
                    'tau': None,
                    'p_value': None,
                    'n_models': len(metric_values)
                })
            else:
                # Compute Kendall's tau
                tau, p_value = kendalltau(human_values, metric_values)
                print(f"    τ = {tau:.4f}, p = {p_value:.4f}")
                
                results.append({
                    'metric': metric_name,
                    'aspect': aspect,
                    'tau': tau,
                    'p_value': p_value,
                    'n_models': len(metric_values)
                })
    
    return results

def save_results(results, model_human_scores, model_metric_scores):
    """Save correlation results and model averages"""
    os.makedirs("./output", exist_ok=True)
    
    # Save correlation results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv("./output/model_level_correlations.csv", index=False)
        print(f"\nCorrelation results saved to ./output/model_level_correlations.csv")
        
        # Create correlation table (pivot format)
        valid_results = df_results[df_results['tau'].notna()]
        if not valid_results.empty:
            correlation_table = valid_results.pivot_table(
                values='tau', 
                index='metric', 
                columns='aspect', 
                aggfunc='mean'
            ).round(4)
            
            # Add average column
            correlation_table['Average'] = correlation_table.mean(axis=1).round(4)
            correlation_table = correlation_table.sort_values('Average', ascending=False)
            
            correlation_table.to_csv('./output/model_level_correlation_table.csv')
            print(f"Correlation table saved to ./output/model_level_correlation_table.csv")
            
            # Print formatted table
            print(f"\nMODEL-LEVEL CORRELATION TABLE (Kendall's Tau):")
            print("=" * 80)
            print(f"{'Metric':<15}", end="")
            for col in correlation_table.columns:
                print(f"{col:>12}", end="")
            print()
            print("-" * 80)
            
            for metric, row in correlation_table.iterrows():
                print(f"{metric:<15}", end="")
                for value in row:
                    if pd.isna(value):
                        print(f"{'N/A':>12}", end="")
                    else:
                        print(f"{value:>12.4f}", end="")
                print()
    
    # Save model averages for inspection
    common_models = set(model_human_scores.keys()) & set(model_metric_scores.keys())
    model_summary = []
    
    for model_id in sorted(common_models):
        row = {'model_id': model_id}
        # Add human scores
        for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
            row[f'human_{aspect}'] = model_human_scores[model_id][aspect]
        # Add metric scores
        for metric in model_metric_scores[model_id].keys():
            row[f'metric_{metric}'] = model_metric_scores[model_id][metric]
        model_summary.append(row)
    
    if model_summary:
        df_summary = pd.DataFrame(model_summary)
        df_summary.to_csv("./output/model_level_averages.csv", index=False)
        print(f"Model averages saved to ./output/model_level_averages.csv")

def main():
    print("MODEL-LEVEL KENDALL'S TAU CORRELATION ANALYSIS")
    print("=" * 60)
    print("Computing correlations between automatic metrics and human annotations")
    print("using model-level averages (12 models)\n")
    
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Compute model-level human scores
    print("Computing model-level human score averages...")
    model_human_scores = compute_model_level_human_scores(data)
    
    print(f"Found {len(model_human_scores)} models with human annotations:")
    for model_id in sorted(model_human_scores.keys()):
        scores = model_human_scores[model_id]
        print(f"  {model_id}: coherence={scores['coherence']:.3f}, "
              f"consistency={scores['consistency']:.3f}, "
              f"fluency={scores['fluency']:.3f}, "
              f"relevance={scores['relevance']:.3f}")
    
    # Load automatic metrics
    print("\nLoading automatic metric scores...")
    metrics_data = load_automatic_metrics()
    
    # if metrics_data['baseline'] is None and metrics_data['ndcg'] is None:
    #     print("Error: No metric files found!")
    #     return
    if metrics_data['ndcg'] is None:
        print("Error: No nDCG metric files found!")
        print("Please run src/ndcg_abs.py to generate nDCG scores")
        return
    
    # Compute model-level metric scores
    print("Computing model-level metric score averages...")
    model_metric_scores = compute_model_level_metric_scores(metrics_data, data)
    
    print(f"Found {len(model_metric_scores)} models with metric scores")
    
    # Compute correlations
    print("Computing model-level Kendall's tau correlations...")
    results = compute_model_level_correlations(model_human_scores, model_metric_scores)
    
    # Save results
    save_results(results, model_human_scores, model_metric_scores)
    
    print(f"Analysis complete! Check ./output/ for:")
    print("  - model_level_correlations.csv (detailed results)")
    print("  - model_level_correlation_table.csv (formatted table)")
    print("  - model_level_averages.csv (model averages used)")
    print(f"\nMetrics included:")
    # if metrics_data['baseline'] is not None:
    #     baseline_metrics = [col for col in metrics_data['baseline'].columns if col != 'model_id']
    #     print(f"  - Baseline: {', '.join(baseline_metrics)}")
    if metrics_data['ndcg'] is not None:
        ndcg_metrics = list(metrics_data['ndcg'].columns)
        print(f"  - nDCG: {', '.join(ndcg_metrics)}")

if __name__ == "__main__":
    main()
