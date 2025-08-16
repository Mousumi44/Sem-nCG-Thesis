import json
import numpy as np
from scipy.stats import kendalltau
import pandas as pd

def compute_kendall_tau_correlations():
    """
    Compute Kendall's tau correlations between model scores and human annotations
    """
    print("Computing Kendall's tau correlations...")

    try:
        # Load human annotations
        with open("./data/processed_data.json", "r", encoding="utf-8") as f:
            samples = json.load(f)
        
        human_scores = {"coherence": [], "consistency": [], "fluency": [], "relevance": []}
        
        for entry in samples:
            if "expert_annotations" in entry and entry["expert_annotations"]:
                anns = entry["expert_annotations"]
                for key in human_scores:
                    vals = [ann.get(key, np.nan) for ann in anns]
                    avg = np.nanmean(vals) if vals else np.nan
                    human_scores[key].append(avg)
            else:
                for key in human_scores:
                    human_scores[key].append(np.nan)

        # Load model scores from CSV files
        score_files = {
            'nDCG': './output/nDCG_scores.csv',
            'ROUGE_BERT': './output/rouge_bert_scores.csv'
        }
        
        results = []
        
        for score_type, file_path in score_files.items():
            try:
                df_scores = pd.read_csv(file_path)
                print(f"\nProcessing {score_type} scores from {file_path}")
                
                for model_col in df_scores.columns:
                    model_scores = df_scores[model_col].dropna().tolist()
                    
                    for annotation_type in human_scores:
                        # Remove pairs where human score is nan and ensure we have matching indices
                        valid = [(h, model_scores[i]) for i, h in enumerate(human_scores[annotation_type]) 
                                if not np.isnan(h) and i < len(model_scores)]
                        
                        if len(valid) < 2:
                            print(f"  {model_col} vs {annotation_type}: Not enough data ({len(valid)} valid pairs)")
                            results.append({
                                "score_type": score_type,
                                "model": model_col, 
                                "annotation": annotation_type, 
                                "tau": None, 
                                "p": None,
                                "valid_pairs": len(valid)
                            })
                            continue
                            
                        h_vals, s_vals = zip(*valid)
                        tau, p = kendalltau(h_vals, s_vals)
                        print(f"  {model_col} vs {annotation_type}: tau={tau:.3f}, p={p:.3g} ({len(valid)} pairs)")
                        results.append({
                            "score_type": score_type,
                            "model": model_col, 
                            "annotation": annotation_type, 
                            "tau": tau, 
                            "p": p,
                            "valid_pairs": len(valid)
                        })
                        
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping {score_type} scores.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Save Kendall's tau results to CSV
        if results:
            df_kendall = pd.DataFrame(results)
            df_kendall.to_csv("./output/kendall_tau_ndcg_results.csv", index=False)
            print(f"\nKendall's tau results saved to ./output/kendall_tau_ndcg_results.csv")
            
            # Print summary
            print("\nSummary of correlations:")
            for score_type in df_kendall['score_type'].unique():
                subset = df_kendall[df_kendall['score_type'] == score_type]
                avg_tau = subset['tau'].mean()
                print(f"  {score_type}: Average tau = {avg_tau:.3f}")
        else:
            print("No results to save.")
        
    except FileNotFoundError:
        print("Error: ./data/processed_data.json not found. Cannot compute Kendall's tau.")
    except Exception as e:
        print(f"Error computing Kendall's tau: {e}")

if __name__ == "__main__":
    compute_kendall_tau_correlations()
