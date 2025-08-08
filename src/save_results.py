import json
import jsonlines
import numpy as np
from scipy.stats import kendalltau
import pandas as pd

# Read the JSONL file
rows = []
with open('./output/score.jsonl', 'r') as f:
    for line in f:
        row = json.loads(line)
        rows.append(row)

# Flatten the data for CSV/Excel
data = {}
for row in rows:
    for key, values in row.items():
        data[key] = values

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

# Save as CSV
df.to_csv('./output/score.csv', index=False)

# --- Compute Kendall Tau ---
sample_file = "./data/processed_data.json"
human_scores = {"coherence": [], "consistency": [], "fluency": [], "relevance": []}

with open(sample_file, "r", encoding="utf-8") as f:
    samples = json.load(f)
    for entry in samples:
        if "expert_annotations" in entry and entry["expert_annotations"]:
            anns = entry["expert_annotations"]
            for key in human_scores:
                # Compute average for this key across all expert annotations
                vals = [ann.get(key, np.nan) for ann in anns]
                avg = np.nanmean(vals) if len(vals) > 0 else np.nan
                human_scores[key].append(avg)
        else:
            for key in human_scores:
                human_scores[key].append(np.nan)

score_file = "./output/score.jsonl"
model_scores = {}
with open(score_file, "r", encoding="utf-8") as f:
    for obj in jsonlines.Reader(f):
        for model, scores in obj.items():
            model_scores[model] = scores

print("Kendall's tau between average human annotation and Sem-nCG scores:")
results = []
for model, scores in model_scores.items():
    print(f"\nModel: {model}")
    for key in human_scores:
        # Remove pairs where human score is nan
        valid = [(h, s) for h, s in zip(human_scores[key], scores) if not np.isnan(h)]
        if not valid or len(valid) < 2:
            print(f"  {key}: Not enough data")
            results.append({"model": model, "annotation": key, "tau": None, "p": None})
            continue
        h_vals, s_vals = zip(*valid)
        tau, p = kendalltau(h_vals, s_vals)
        print(f"  {key}: tau={tau:.3f}, p={p:.3g}")
        results.append({"model": model, "annotation": key, "tau": tau, "p": p})

# Save results to CSV
df_kendall = pd.DataFrame(results)
df_kendall.to_csv("./output/kendall_results.csv", index=False)
print("Kendall's tau results saved to ./output/kendall_results.csv.")
