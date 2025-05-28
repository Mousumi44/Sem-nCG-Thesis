import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
from nltk import tokenize

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def doc_per_sent_similarity(doc_sent_embeddings, summary_sent_embeddings):
    doc_sent_similarity = {}
    sen_id = 0
    for e1 in doc_sent_embeddings:
        e2_cos_sum = 0
        for e2 in summary_sent_embeddings:
            e1_cpu = e1.detach().cpu().numpy().reshape(1, -1)
            e2_cpu = e2.detach().cpu().numpy().reshape(1, -1)
            e2_cos_sum += cosine_similarity(e1_cpu, e2_cpu)[0][0]
        doc_sent_similarity[sen_id] = e2_cos_sum / summary_sent_embeddings.shape[0]
        sen_id += 1
    return doc_sent_similarity

def compute_similarity(doc_sent, ref_sent, tokenizer, model):
    device = next(model.parameters()).device
    encoded_doc = tokenizer(doc_sent, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    encoded_ref = tokenizer(ref_sent, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_doc = model(**encoded_doc)
        model_ref = model(**encoded_ref)
    doc_sent_embeddings = mean_pooling(model_doc, encoded_doc['attention_mask'])
    ref_sent_embeddings = mean_pooling(model_ref, encoded_ref['attention_mask'])
    return doc_per_sent_similarity(doc_sent_embeddings, ref_sent_embeddings)

def load_openelm_and_tokenizer(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer, model

if __name__ == "__main__":
    model_path = "apple/OpenELM-270M"
    print(f"[INFO] Loading OpenELM model and compatible tokenizer: {model_path}")
    tokenizer, model = load_openelm_and_tokenizer(model_path)
    print("[INFO] Model and tokenizer loaded.")

    # Example: process a sample file as in pre-run.py
    with open("sample.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    doc = df["Doc"]
    ref = df["Reference"]

    for i in range(len(doc)):
        doc_sent = tokenize.sent_tokenize(doc[i])
        ref_sent = tokenize.sent_tokenize(ref[i])
        sim = compute_similarity(doc_sent, ref_sent, tokenizer, model)
        sim = {k: float(v) for k, v in sim.items()}
        print(f"Sample {i} similarity:", sim)