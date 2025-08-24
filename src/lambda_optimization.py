#!/usr/bin/env python3
"""
Lambda Parameter Optimization for nDCG-abs
Tests different lambda values to find optimal weighting between nDCG and cosine similarity
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import sys
import os

# Add src to path to import ndcg_abs
sys.path.append('./src')
from ndcg_abs import NDCGAbsEvaluator

def test_lambda_values(lambda_values, model_name="sbert-mini"):
    """
    Test different lambda values and compute MODEL-LEVEL correlations with human judgments
    
    Args:
        lambda_values: List of lambda values to test
        model_name: Embedding model to use
    """
    print(f"Testing lambda values: {lambda_values}")
    print(f"Using embedding model: {model_name}")
    print(f"Computing MODEL-LEVEL correlations (12 models)")
    print("=" * 60)
    
    # Load data
    with open("./data/processed_data.json", "r") as f:
        data = json.load(f)
    
    # Filter to abstractive summaries
    test_data = [item for item in data if item.get("summary_type") == "Abstractive"]
    print(f"Loaded {len(test_data)} test samples")
    
    # Get model path from name
    from ndcg_abs import get_model_path_from_name
    model_path = get_model_path_from_name(model_name)
    
    # Initialize evaluator
    evaluator = NDCGAbsEvaluator(model_name=model_path)
    
    results = {}
    
    for lambda_val in lambda_values:
        print(f"\nTesting λ = {lambda_val}")
        
        # Create new evaluator with this lambda value to ensure it's properly set
        evaluator = NDCGAbsEvaluator(model_name=model_path)
        evaluator.lambda_param = lambda_val
        
        # Debug: verify lambda is set correctly
        print(f"  Evaluator lambda_param: {evaluator.lambda_param}")
        
        # Compute scores for all samples and group by model
        model_scores = {}
        model_human_scores = {}
        
        for i, item in enumerate(test_data):
            try:
                model_id = item.get('model_id', 'unknown')
                
                # Initialize model tracking
                if model_id not in model_scores:
                    model_scores[model_id] = {
                        'ndcg_abs': [], 'ndcg': [], 'cosine': []
                    }
                    model_human_scores[model_id] = {
                        'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []
                    }
                
                # Compute nDCG-abs with current lambda
                result = evaluator.compute_ndcg_abs(
                    item["Doc"], 
                    item["Reference"], 
                    item["model"]
                )
                
                model_scores[model_id]['ndcg_abs'].append(result['ndcg_abs'])
                model_scores[model_id]['ndcg'].append(result['ndcg'])
                model_scores[model_id]['cosine'].append(result['cosine_score'])
                
                # Get human scores (average across annotators for this sample)
                if "expert_annotations" in item and item["expert_annotations"]:
                    annotations = item["expert_annotations"]
                    
                    for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
                        aspect_scores = [ann.get(aspect) for ann in annotations if ann.get(aspect) is not None]
                        if aspect_scores:
                            model_human_scores[model_id][aspect].append(np.mean(aspect_scores))
                        else:
                            model_human_scores[model_id][aspect].append(np.nan)
                else:
                    for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
                        model_human_scores[model_id][aspect].append(np.nan)
                    
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        # Average per model (MODEL-LEVEL aggregation)
        model_avg_scores = {}
        model_avg_human = {}
        
        for model_id in model_scores:
            # Average automatic scores per model
            model_avg_scores[model_id] = {
                'ndcg_abs': np.mean(model_scores[model_id]['ndcg_abs']),
                'ndcg': np.mean(model_scores[model_id]['ndcg']),
                'cosine': np.mean(model_scores[model_id]['cosine'])
            }
            
            # Average human scores per model
            model_avg_human[model_id] = {}
            for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
                valid_scores = [s for s in model_human_scores[model_id][aspect] if not np.isnan(s)]
                model_avg_human[model_id][aspect] = np.mean(valid_scores) if valid_scores else np.nan
        
        # Compute MODEL-LEVEL correlations (12 data points)
        correlations = {}
        
        # Debug: Print some sample scores to verify lambda is working
        if len(model_avg_scores) > 0:
            sample_model = list(model_avg_scores.keys())[0]
            sample_scores = model_avg_scores[sample_model]
            print(f"  Sample scores for {sample_model}: nDCG={sample_scores['ndcg']:.3f}, "
                  f"cosine={sample_scores['cosine']:.3f}, nDCG-abs={sample_scores['ndcg_abs']:.3f}")
            
            # Verify lambda calculation manually
            expected_ndcg_abs = lambda_val * sample_scores['ndcg'] + (1 - lambda_val) * sample_scores['cosine']
            print(f"  Expected nDCG-abs: {expected_ndcg_abs:.3f}, Actual: {sample_scores['ndcg_abs']:.3f}")
        
        if len(model_avg_scores) >= 3:  # Need minimum models for correlation
            try:
                # Get arrays of model averages
                models = sorted(model_avg_scores.keys())
                
                ndcg_abs_values = [model_avg_scores[m]['ndcg_abs'] for m in models]
                ndcg_values = [model_avg_scores[m]['ndcg'] for m in models]
                cosine_values = [model_avg_scores[m]['cosine'] for m in models]
                
                for aspect in ['coherence', 'consistency', 'fluency', 'relevance']:
                    human_values = [model_avg_human[m][aspect] for m in models]
                    
                    # Remove models with NaN human scores
                    valid_indices = [i for i, h in enumerate(human_values) if not np.isnan(h)]
                    
                    if len(valid_indices) >= 3:
                        valid_ndcg_abs = [ndcg_abs_values[i] for i in valid_indices]
                        valid_ndcg = [ndcg_values[i] for i in valid_indices]
                        valid_cosine = [cosine_values[i] for i in valid_indices]
                        valid_human = [human_values[i] for i in valid_indices]
                        
                        correlations[aspect] = kendalltau(valid_human, valid_ndcg_abs)[0]
                        correlations[f'ndcg_{aspect}'] = kendalltau(valid_human, valid_ndcg)[0]
                        correlations[f'cosine_{aspect}'] = kendalltau(valid_human, valid_cosine)[0]
                    else:
                        correlations[aspect] = 0.0
                        correlations[f'ndcg_{aspect}'] = 0.0
                        correlations[f'cosine_{aspect}'] = 0.0
                
                correlations['average'] = np.mean([correlations[asp] for asp in ['coherence', 'consistency', 'fluency', 'relevance']])
                correlations['n_models'] = len(valid_indices) if 'valid_indices' in locals() else len(models)
                
            except Exception as e:
                print(f"Error computing correlations: {e}")
                correlations = {k: 0.0 for k in ['coherence', 'consistency', 'fluency', 'relevance', 'average']}
                correlations['n_models'] = 0
        else:
            correlations = {k: 0.0 for k in ['coherence', 'consistency', 'fluency', 'relevance', 'average']}
            correlations['n_models'] = len(model_avg_scores)
        
        results[lambda_val] = correlations
        
        print(f"  Correlations ({correlations['n_models']} models): coherence={correlations['coherence']:.3f}, "
              f"consistency={correlations['consistency']:.3f}, "
              f"fluency={correlations['fluency']:.3f}, "
              f"relevance={correlations['relevance']:.3f}")
        print(f"  Average: {correlations['average']:.3f}")
    
    return results

def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "=" * 90)
    print("LAMBDA OPTIMIZATION RESULTS (MODEL-LEVEL CORRELATIONS)")
    print("=" * 90)
    
    # Header
    print(f"{'Lambda':<8} {'Coherence':<10} {'Consistency':<12} {'Fluency':<8} {'Relevance':<9} {'Average':<8} {'N_Models':<8}")
    print("-" * 90)
    
    # Sort by average correlation
    sorted_results = sorted(results.items(), key=lambda x: x[1]['average'], reverse=True)
    
    for lambda_val, corrs in sorted_results:
        n_models = corrs.get('n_models', 0)
        print(f"{lambda_val:<8.2f} {corrs['coherence']:<10.3f} {corrs['consistency']:<12.3f} "
              f"{corrs['fluency']:<8.3f} {corrs['relevance']:<9.3f} {corrs['average']:<8.3f} {n_models:<8}")
    
    # Find best lambda
    best_lambda = sorted_results[0][0]
    best_avg = sorted_results[0][1]['average']
    n_models = sorted_results[0][1].get('n_models', 0)
    
    print("\n" + "=" * 90)
    print(f"RECOMMENDATION: Use λ = {best_lambda:.2f} (Average correlation: {best_avg:.3f}, {n_models} models)")
    print("=" * 90)

if __name__ == "__main__":
    # Test with fewer lambda values first for debugging
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = test_lambda_values(lambda_values)
    print_results_table(results)
    
    # Save results
    df_results = pd.DataFrame(results).T
    df_results.to_csv("./output/lambda_optimization_results.csv")
    print(f"\nResults saved to ./output/lambda_optimization_results.csv")
