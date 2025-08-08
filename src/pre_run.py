import re
from re import search
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
from nltk import tokenize
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaTokenizer
from sentence_transformers import SentenceTransformer
# from laser_encoders import LaserEncoderPipeline
import sys
import ast
from copy import deepcopy
from dotenv import load_dotenv
import tensorflow_hub as hub
import tensorflow as tf
import difflib


# import nltk
# nltk.download('punkt_tab')

load_dotenv() 
token = os.getenv("HF_TOKEN")  

from huggingface_hub import login
login(token=token)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import torch

# LLM Models and Classical Models to be used for computing sentence embeddings
LLM_MODELS = [

    # LLMs (uncomment as needed)
    
    # ('meta-llama/Llama-3.2-1B', 'llama3.2'),
    # ("meta-llama/Llama-2-13b-hf",'llama2'),
    # ('google/gemma-3-1b-it', 'gemma-3-1b-it'),
    # ('mistralai/Mistral-7B-v0.1', 'mistral'),
    # ("apple/OpenELM-270M", "openelm"),
    # ("allenai/OLMo-2-0425-1B", "olmo-2-1b"),
    # ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    # ('AtlaAI/Selene-1-Mini-Llama-3.1-8B', 'selene-llama3.1-8b'),
    # ('tiiuae/falcon-7B', 'falcon-7b'),
    # ('microsoft/phi-4', 'phi-4'),
    # ('microsoft/Phi-3-mini-instruct', 'phi-3-mini-instruct'),
    ('deepseek-ai/DeepSeek-V2.5', 'deepseek-v2.5'),
    # ('yulan-team/YuLan-Mini', 'yulan-mini'),

    # Classical models
    # ('sentence-transformers/all-MiniLM-L6-v2', 'sbert-mini'),
    # ('sentence-transformers/all-mpnet-base-v2', 'sbert-l'),
    # ('laserembeddings', 'laser'),
    # ('universal-sentence-encoder', 'use'),
    # ('roberta-base', 'roberta'),
    # ('princeton-nlp/sup-simcse-roberta-base', 'simcse'),
    # ('InferSent/encoder/infersent2.pkl', 'infersent'),
]

def load_model(model_path):    
    if model_path == "universal-sentence-encoder":
        # For Universal Sentence Encoder, load from TF Hub
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        # Return model as both tokenizer and model since USE handles both functions
        return model, model
    elif "openelm" in model_path.lower():
        # Use LlamaTokenizer as a compatible tokenizer for OpenELM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer, model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        return tokenizer, model

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def doc_per_sent_similarity(doc_sent_embeddings, summary_sent_embeddings):
  doc_sent_similarity = {}
  sen_id = 0

  for e1 in doc_sent_embeddings:
    e2_cos_sum = 0
    for e2 in summary_sent_embeddings:
      # Ensure tensors are on CPU before converting to numpy
      e1_cpu = e1.detach().cpu().numpy().reshape(1, -1)
      e2_cpu = e2.detach().cpu().numpy().reshape(1, -1)
      e2_cos_sum += cosine_similarity(e1_cpu, e2_cpu)[0][0]
    doc_sent_similarity[sen_id] = e2_cos_sum / summary_sent_embeddings.shape[0]
    sen_id += 1

  return doc_sent_similarity


def compute_similarity(doc_sent, ref_sent, tokenizer, model):
    # For USE model - it handles both tokenization and embedding
    if hasattr(model, '_is_hub_module_v1'):
        doc_sent_embeddings = model(tf.constant(doc_sent))
        ref_sent_embeddings = model(tf.constant(ref_sent))
        return doc_per_sent_similarity(torch.from_numpy(doc_sent_embeddings.numpy()), 
                                    torch.from_numpy(ref_sent_embeddings.numpy()))

    # For other models that use PyTorch
    device = next(model.parameters()).device
    encoded_doc = tokenizer(doc_sent, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    encoded_ref = tokenizer(ref_sent, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_doc = model(**encoded_doc)
        model_ref = model(**encoded_ref)
    doc_sent_embeddings = mean_pooling(model_doc, encoded_doc['attention_mask'])
    ref_sent_embeddings = mean_pooling(model_ref, encoded_ref['attention_mask'])
    return doc_per_sent_similarity(doc_sent_embeddings, ref_sent_embeddings)

def read_sample_file(file="./data/processed_data.json", summary_type=None):
    """
    :param summary_type: "Extractive", "Abstractive", or None for all
    :rtype: sample.json -> [{"Doc": str, "Reference": str, "model": str},...], Doc->List[str], Reference->List[str], model->List[str]
    """
    data = json.load(open(file, "r"))
    if summary_type:
        data = [d for d in data if d.get("summary_type") == summary_type]
    df = pd.DataFrame.from_dict(data)
    return df["Doc"], df["Reference"], df["model"]


# Improved version for extractive summaries: lower, remove punctuation, collapse whitespace, strip, and match
# Might need to adjust for abstractive summaries
def compute_senID(doc_sent, model_summary_sent):
    import string
    doc_sents = deepcopy(doc_sent)
    model_sum_SI = []
    seen_doc_sent_idxs = []
    def clean(s):
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = re.sub(r"\s+", " ", s)
        return s.strip()
    cleaned_doc_sents = [clean(s) for s in doc_sents]
    for search in model_summary_sent:
        search_clean = clean(search)
        for doc_idx, s in enumerate(cleaned_doc_sents):
            if search_clean == s and doc_idx not in seen_doc_sent_idxs:
                model_sum_SI.append(doc_idx)
                seen_doc_sent_idxs.append(doc_idx)
                break
        # else:
        #     # fallback: substring match
        #     for doc_idx, s in enumerate(cleaned_doc_sents):
        #         if search_clean in s and doc_idx not in seen_doc_sent_idxs:
        #             model_sum_SI.append(doc_idx)
        #             seen_doc_sent_idxs.append(doc_idx)
        #             break
    if len(model_sum_SI) < len(model_summary_sent):
        print(f"[SKIP] matched {len(model_sum_SI)} of {len(model_summary_sent)}")
        return None
    return model_sum_SI


def compute_model_senId(doc, model, summary_type=None):
    """
    :type model: List[str]
    :param summary_type: "Extractive" or "Abstractive" or None
    :rtype: List[List[int]] -> write model and sen_Id to model_<summary_type>.json file
    """
    modelSenId = []
    for i in range(len(model)):
        doc_sent = tokenize.sent_tokenize(doc[i])
        model_summary_sent = tokenize.sent_tokenize(model[i])
        model_sum_SI = compute_senID(doc_sent, model_summary_sent)
        modelSenId.append(model_sum_SI)
    df = pd.DataFrame(
        {'model': model,
         'modelSenID': modelSenId
        })
    if summary_type:
        out_path = f"./models/model_{summary_type.lower()}.json"
    else:
        out_path = "./models/model.json"
    df.to_json(out_path, orient='records') #orient="records"/"index"
    print(f"[INFO] Saved model sentence IDs to {out_path}")
    return


def compute_doc_ref_similarity():
    dir = "output"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Set summary_types here:

    summary_types = ["Extractive"]
    # summary_types = ["Abstractive"]
    # summary_types = ["Extractive", "Abstractive"]  #both

    for summary_type in summary_types:
        print(f"[INFO] Processing summary_type: {summary_type}")
        doc, ref, model = read_sample_file(summary_type=summary_type)
        compute_model_senId(doc, model, summary_type=summary_type)

        for model_path, model_name in LLM_MODELS:
            print(f"[INFO] Running model: {model_name} ({model_path}) [{summary_type}]")
            fn = f"./models/{model_name}_similarity_{summary_type.lower()}.txt"
            tokenizer, model_obj = load_model(model_path)
            with open(fn, "w") as fw:
                for i in range(len(doc)):
                    doc_sent = tokenize.sent_tokenize(doc[i])
                    ref_sent = tokenize.sent_tokenize(ref[i])
                    sim = compute_similarity(doc_sent, ref_sent, tokenizer, model_obj)
                    sim = {k: float(v) for k, v in sim.items()}
                    json.dump(sim, fw)
                    fw.write("\n")
            print(f"[INFO] Saved similarity scores to {fn}")


def read_file(model_name, summary_type=None):
    """
    #read corresponding <model_name>_similarity[_summary_type].txt file#
    :param summary_type: "Extractive", "Abstractive", or None
    :rtype Dataframe: fileId, dictSim -> sim of doc sentences with reference
    """
    dir = "./models/"
    if summary_type:
        fn = dir+f"{model_name}_similarity_{summary_type.lower()}.txt"
    else:
        fn = dir+f"{model_name}_similarity.txt"

    dicSim = []
    with open(fn, "r") as file:
        for line in file:
            dicSim.append(json.loads(line))

    fileId = [i for i in range(len(dicSim))]
    df = pd.DataFrame(
        {"Idx" : fileId,
        "DictSim": dicSim
        })
    return df

def computeGain(dic_tuple):
    """
    :type dic_tuple: List[tuple] -> already sorted based on score
    :rtype: List[tuple] -> new_dic_tuple -> key:sid, value:gain
    """
    denom = 1.0*sum([i+1 for i in range(len(dic_tuple))])

    new_dic = {}
    g = len(dic_tuple)
    for sId, score in dic_tuple:
        new_dic[sId] = (g*1.0)/denom
        g-=1
    return sorted(new_dic.items(), key = lambda i : i[1], reverse=True)



def compute_gt():
    """
    Write output to file for each model and summary_type
    """
    dir = "./models/"
    # Set summary_types here:
    summary_types = ["Extractive"]
    # summary_types = ["Abstractive"]
    # summary_types = ["Abstractive", "Extractive"]  # both

    for summary_type in summary_types:
        for _, model_name in LLM_MODELS:
            fn = dir+f"{model_name}_gain_{summary_type.lower()}.txt"
            try:
                df = read_file(model_name, summary_type=summary_type)
            except FileNotFoundError:
                print(f"[SKIP] {model_name} {summary_type}: similarity file not found, skipping gain computation.")
                continue
            samples = len(df)
            with open(fn, "w") as fw:
                for i in range(samples):
                    prev = sorted(df["DictSim"][i].items(), key=lambda i: i[1], reverse=True)
                    fw.write(str(computeGain(prev)))
                    fw.write("\n")
                    print(f'[{model_name}][{summary_type}] Current Sample {i}')



if __name__=="__main__":
    
    compute_doc_ref_similarity()
    compute_gt()