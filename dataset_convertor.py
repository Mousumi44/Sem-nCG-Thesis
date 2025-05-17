import json

# File paths
input_file = "data_annotations_aligned.paired.jsonl"
output_file = "prepared_data.json"

converted_data = []

converted = []

with open(input_file, "r", encoding="latin1") as f:
    for line in f:
        try:
            entry = json.loads(line)

            # Ensure required fields exist and are not empty
            if "text" in entry and "decoded" in entry and "references" in entry and entry["references"]:
                converted.append({
                    "Doc": entry["text"],
                    "Reference": entry["references"][0],  # use all sentences in references?
                    "model": entry["decoded"]
                })

        except json.JSONDecodeError:
            print("Skipping malformed line.")

# Write to Sem-nCG input format
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print(f"Converted {len(converted)} samples to Sem-nCG format.")