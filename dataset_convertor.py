
import json
import unicodedata
import re

# File paths
input_file = "data_annotations_aligned.paired.jsonl"
output_file = "prepared_data_v2.json"

converted_data = []

converted = []


# Cleaning function to ensure ASCII output and replace unicode punctuation
def clean_to_ascii(text):
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',  # non-breaking space
        '\u2010': '-',  # hyphen
        '\u2012': '-',  # figure dash
        '\u2015': '-',  # horizontal bar
        '\u2212': '-',  # minus sign
        '\u00b7': ' ',  # middle dot
        '\u200b': '',   # zero width space
        '\u2011': '-',  # non-breaking hyphen
        '\u00ab': '"',  # left angle quote
        '\u00bb': '"',  # right angle quote
        '\u02c6': '^',  # modifier letter circumflex accent
        '\u2039': '<',  # single left-pointing angle quotation mark
        '\u203a': '>',  # single right-pointing angle quotation mark
        '\u2022': '*',  # bullet
        '\u2122': 'TM', # trademark
        '\u00e9': 'e',  # e with acute
    }
    # Replace unicode escapes (if present as literal)
    for uni, ascii_char in replacements.items():
        text = text.replace(uni, ascii_char)
    # Replace actual unicode characters
    for uni, ascii_char in replacements.items():
        try:
            text = text.replace(uni.encode('utf-8').decode('unicode_escape'), ascii_char)
        except Exception:
            pass
    # Normalize and remove any remaining non-ASCII
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

with open(input_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            entry = json.loads(line)
            # Ensure required fields exist and are not empty
            if "text" in entry and "decoded" in entry and "references" in entry and entry["references"]:
                doc = clean_to_ascii(entry["text"])
                reference = clean_to_ascii(entry["references"][0])  # Assuming first reference is the one to use
                model = clean_to_ascii(entry["decoded"])
                converted.append({
                    "Doc": doc,
                    "Reference": reference,
                    "model": model
                })
        except json.JSONDecodeError:
            print("Skipping malformed line.")

# Write to Sem-nCG input format
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print(f"Converted {len(converted)} samples to Sem-nCG format.")