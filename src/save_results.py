import json
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

# Save as Excel
# df.to_excel('./output/score.xlsx', index=False)