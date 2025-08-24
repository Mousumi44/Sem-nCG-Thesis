import json
import numpy as np
from scipy.stats import kendalltau
import pandas as pd

def compute_kendall_tau_correlations():
    """
    Compute Kendall's tau correlations between model scores and human annotations
    Model-level correlation: 12 models × 100 documents = 1200 abstractive summaries
    """
    print("Computing Kendall's tau correlations (model-level)...")

    try:
        # Load human annotations
        with open("./data/processed_data.json", "r", encoding="utf-8") as f:
            samples = json.load(f)
        
        # Filter to abstractive summaries only
        abstractive_samples = [s for s in samples if s.get('summary_type') == 'Abstractive']
        print(f"Found {len(abstractive_samples)} abstractive summaries")
        
        # Group by model_id and compute average human scores per model
        model_human_scores = {}
        
        for item in abstractive_samples:
            model_id = item.get('model_id', 'unknown')
            
            if model_id not in model_human_scores:
                model_human_scores[model_id] = {
                    'coherence': [], 'consistency': [], 'fluency': [], 'relevance': []
                }
            
            # Average the multiple human annotations for this summary
            if "expert_annotations" in item and item["expert_annotations"]:
                anns = item["expert_annotations"]
                for annotation_type in ['coherence', 'consistency', 'fluency', 'relevance']:
                    # Get all scores for this annotation type and average them
                    scores = [ann.get(annotation_type, np.nan) for ann in anns if annotation_type in ann]
                    if scores:
                        avg_score = np.nanmean(scores)
                        model_human_scores[model_id][annotation_type].append(avg_score)
                    else:
                        model_human_scores[model_id][annotation_type].append(np.nan)
            else:
                # No annotations for this summary
                for annotation_type in ['coherence', 'consistency', 'fluency', 'relevance']:
                    model_human_scores[model_id][annotation_type].append(np.nan)
        
        # Average across all summaries for each model
        model_avg_human_scores = {}
        for model_id, scores_dict in model_human_scores.items():
            model_avg_human_scores[model_id] = {}
            for annotation_type, scores in scores_dict.items():
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    model_avg_human_scores[model_id][annotation_type] = np.mean(valid_scores)
                else:
                    model_avg_human_scores[model_id][annotation_type] = np.nan
        
        print(f"Found {len(model_avg_human_scores)} models with human scores:")
        for model_id in sorted(model_avg_human_scores.keys()):
            scores = model_avg_human_scores[model_id]
            print(f"  {model_id}: coherence={scores['coherence']:.3f}, consistency={scores['consistency']:.3f}, "
                  f"fluency={scores['fluency']:.3f}, relevance={scores['relevance']:.3f}")

        # Load model scores from CSV files
        score_files = {
            'nDCG': './output/nDCG_scores.csv',
            'Baseline_Models': './output/baseline_models_scores.csv'
        }
        
        results = []
        
        for score_type, file_path in score_files.items():
            try:
                df_scores = pd.read_csv(file_path)
                print(f"\nProcessing {score_type} scores from {file_path}")
                print(f"Score file shape: {df_scores.shape}")
                print(f"Metrics available: {list(df_scores.columns)}")
                
                # For each metric column in the CSV
                for metric_name in df_scores.columns:
                    print(f"\n  Processing metric: {metric_name}")
                    
                    # Get all scores for this metric (should be 1200 scores)
                    metric_scores = df_scores[metric_name].dropna().tolist()
                    print(f"    Total scores: {len(metric_scores)}")
                    
                    if len(metric_scores) != 1200:
                        print(f"    Warning: Expected 1200 scores, got {len(metric_scores)}")
                        continue
                    
                    # Group scores by model (100 scores per model)
                    # Assuming scores are ordered by model: M0 (100), M1 (100), M2 (100), etc.
                    model_metric_averages = {}
                    
                    # Get sorted model IDs from our data
                    sorted_model_ids = sorted(model_avg_human_scores.keys())
                    
                    for i, model_id in enumerate(sorted_model_ids):
                        start_idx = i * 100
                        end_idx = start_idx + 100
                        
                        if end_idx <= len(metric_scores):
                            model_scores = metric_scores[start_idx:end_idx]
                            model_metric_averages[model_id] = np.mean(model_scores)
                            print(f"    {model_id}: avg = {model_metric_averages[model_id]:.4f}")
                        else:
                            print(f"    Warning: Not enough scores for model {model_id}")
                            break
                    
                    # Now correlate model-level averages
                    for annotation_type in ['coherence', 'consistency', 'fluency', 'relevance']:
                        print(f"\n    Correlating {metric_name} vs {annotation_type}:")
                        
                        model_pairs = []
                        for model_id in sorted_model_ids:
                            if (model_id in model_metric_averages and 
                                model_id in model_avg_human_scores):
                                
                                metric_avg = model_metric_averages[model_id]
                                human_avg = model_avg_human_scores[model_id][annotation_type]
                                
                                if not np.isnan(human_avg) and not np.isnan(metric_avg):
                                    model_pairs.append((human_avg, metric_avg))
                                    print(f"      {model_id}: human={human_avg:.3f}, metric={metric_avg:.4f}")
                        
                        if len(model_pairs) < 3:
                            print(f"      Not enough valid model pairs ({len(model_pairs)}) for correlation")
                            results.append({
                                "score_type": score_type,
                                "metric": metric_name, 
                                "annotation": annotation_type, 
                                "tau": None, 
                                "p": None,
                                "valid_models": len(model_pairs)
                            })
                            continue
                            
                        h_vals, s_vals = zip(*model_pairs)
                        tau, p = kendalltau(h_vals, s_vals)
                        print(f"      Kendall's tau = {tau:.4f}, p = {p:.4f} ({len(model_pairs)} models)")
                        
                        results.append({
                            "score_type": score_type,
                            "metric": metric_name, 
                            "annotation": annotation_type, 
                            "tau": tau, 
                            "p": p,
                            "valid_models": len(model_pairs)
                        })
                        
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping {score_type} scores.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Save Kendall's tau results to CSV
        if results:
            df_kendall = pd.DataFrame(results)
            df_kendall.to_csv("./output/kendall_tau_model_level_results.csv", index=False)
            print(f"\nKendall's tau results saved to ./output/kendall_tau_model_level_results.csv")
            
            # Create correlation table in the same format as compare_models.py expects
            if len(df_kendall) > 0:
                correlation_table = df_kendall.pivot_table(
                    values='tau', 
                    index='metric', 
                    columns='annotation', 
                    aggfunc='mean'
                ).round(3)
                
                # Add average column
                correlation_table['Average'] = correlation_table.mean(axis=1).round(3)
                correlation_table = correlation_table.sort_values('Average', ascending=False)
                
                # Save correlation table
                correlation_table.to_csv('./output/correlation_table.csv')
                print(f"Correlation table saved to ./output/correlation_table.csv")
                
                # Print formatted correlation table
                print(f"\nCORRELATION TABLE (Kendall's Tau):")
                print("="*80)
                header = f"{'Metric':<25}"
                for col in correlation_table.columns:
                    header += f"{col:>12}"
                print(header)
                print("-" * len(header))
                
                for metric, row in correlation_table.iterrows():
                    row_str = f"{metric:<25}"
                    for value in row:
                        if pd.isna(value):
                            row_str += f"{'N/A':>12}"
                        else:
                            row_str += f"{value:>12.3f}"
                    print(row_str)
            
            # Print summary of best correlations
            print(f"\nSUMMARY OF MODEL-LEVEL CORRELATIONS:")
            print("="*60)
            
            for annotation_type in ['coherence', 'consistency', 'fluency', 'relevance']:
                print(f"\n{annotation_type.upper()}:")
                subset = df_kendall[(df_kendall['annotation'] == annotation_type) & 
                                   (df_kendall['tau'].notna())]
                if len(subset) > 0:
                    # Sort by absolute tau value
                    subset_sorted = subset.reindex(subset['tau'].abs().sort_values(ascending=False).index)
                    for _, row in subset_sorted.head(5).iterrows():
                        print(f"  {row['metric']:15s}: τ = {row['tau']:6.3f} (p = {row['p']:.3f})")
                else:
                    print("  No valid correlations found")
                    
        else:
            print("No results to save.")
        
    except FileNotFoundError:
        print("Error: ./data/processed_data.json not found. Cannot compute Kendall's tau.")
    except Exception as e:
        print(f"Error computing Kendall's tau: {e}")

if __name__ == "__main__":
    compute_kendall_tau_correlations()
