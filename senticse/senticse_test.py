import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("DILAB-HYU/SentiCSE")
model = AutoModel.from_pretrained("DILAB-HYU/SentiCSE")

# Tokenize input texts
texts = [
    "The food is delicious.",
    "The atmosphere of the restaurant is good.",
    "The food at the restaurant is devoid of flavor.",
    "The restaurant lacks a good ambiance."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
cosine_sim_0_3 = 1 - cosine(embeddings[0], embeddings[3])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[3], cosine_sim_0_3))