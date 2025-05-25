import re
from re import search
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
from nltk import tokenize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import sys
import ast
from copy import deepcopy
from dotenv import load_dotenv


# import nltk
# nltk.download('punkt_tab')

load_dotenv() 
token = os.getenv("HF_TOKEN")  

from huggingface_hub import login
login(token=token)

# LLM Models to be used for computing sentence embeddings
LLM_MODELS = [
    ('meta-llama/Llama-3.2-1B', 'llama3.2'),
    # ('google/gemma-3-1b-it', 'gemma-3-1b-it'),
    # ('mistralai/Mistral-7B-v0.1', 'mistral'),
    # ("meta-llama/Llama-3.2-1B", "llama3.2-1b"),
    # ("apple/OpenELM-3B", "openelm-3b"),
    # ("allenai/OLMo-7B", "olmo-7b"),
    # ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    # ("meta-llama/Llama-2-13b-hf",'llama2-13b'),
    # ('sentence-transformers/all-MiniLM-L6-v2', 'sbert-mini'),
    # ('sentence-transformers/distilbert-base-uncased', 'distilbert')
]

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModel.from_pretrained(model_path)
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
      e2_cos_sum+=cosine_similarity(e1.reshape(1,-1), e2.reshape(1,-1))[0][0]
    doc_sent_similarity[sen_id] = e2_cos_sum/summary_sent_embeddings.shape[0]
    sen_id +=1

  return doc_sent_similarity


def compute_similarity(doc_sent, ref_sent, tokenizer, model):
    encoded_doc = tokenizer(doc_sent, padding=True, truncation=True, return_tensors='pt')
    encoded_ref = tokenizer(ref_sent, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_doc = model(**encoded_doc)
        model_ref = model(**encoded_ref)
    doc_sent_embeddings = mean_pooling(model_doc, encoded_doc['attention_mask'])
    ref_sent_embeddings = mean_pooling(model_ref, encoded_ref['attention_mask'])
    return doc_per_sent_similarity(doc_sent_embeddings, ref_sent_embeddings)

def read_sample_file(file="sample.json"):
    """
    :rtype: sample.json -> [{"Doc": str, "Reference": str, "model": str},...], Doc->List[str], Reference->List[str], model->List[str]
    """
    data = json.load(open(file, "r"))
    df = pd.DataFrame.from_dict(data)
    return df["Doc"], df["Reference"], df["model"]

def compute_senID(doc_sent, model_summary_sent): # might need to change to compute semantic similarities (for abstraction)
    doc_sents = deepcopy(doc_sent)
    model_sum_SI = []

    seen_doc_sent_idxs = []
    for search in model_summary_sent:
        search = search.lower()
        search = re.sub(r"[^a-zA-Z0-9]", "", search)
        for doc_idx, s in enumerate(doc_sents):
            s = s.lower()
            s = re.sub(r"[^a-zA-Z0-9]", "", s)
            if search in s:
                if doc_idx not in seen_doc_sent_idxs:
                    model_sum_SI.append(doc_idx)
                    seen_doc_sent_idxs.append(doc_idx)
                    break

    # assert len(model_sum_SI) == len(model_summary_sent)
    
    if len(model_sum_SI) < len(model_summary_sent):  # changed due to abstractive summaries
      print(f"[SKIP] Only matched {len(model_sum_SI)} of {len(model_summary_sent)}")
      return None

    return model_sum_SI


def compute_model_senId(doc, model):
  """
  :type model: List[str]
  :rtype: List[List[int]] -> write model and sen_Id to model.json file
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
  df.to_json("./output/model.json", orient='records') #orient="records"/"index"
  return 


def compute_doc_ref_similarity():
    dir = "output"
    if not os.path.exists(dir):
        os.makedirs(dir)
    doc, ref, model = read_sample_file()
    compute_model_senId(doc, model)

    for model_path, model_name in LLM_MODELS:
        fn = f"./output/{model_name}_similarity.txt"
        tokenizer, model_obj = load_model(model_path)
        with open(fn, "w") as fw:
            for i in range(len(doc)):
                doc_sent = tokenize.sent_tokenize(doc[i])
                ref_sent = tokenize.sent_tokenize(ref[i])
                sim = compute_similarity(doc_sent, ref_sent, tokenizer, model_obj)
                sim = {k: float(v) for k, v in sim.items()}
                json.dump(sim, fw)
                fw.write("\n")


def read_file(model_name):
    """
    #read corresponding <model_name>_similarity.txt file#
    :rtype Dataframe: fileId, dictSim -> sim of doc sentences with reference
    """
    dir = "./output/"
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
    write output to file for each model
    """
    dir = "./output/"
    for _, model_name in LLM_MODELS:
        fn = dir+f"{model_name}_gain.txt"
        df = read_file(model_name)
        samples = len(df)
        with open(fn, "w") as fw:
            for i in range(samples):
                prev = sorted(df["DictSim"][i].items(), key=lambda i: i[1], reverse=True)
                fw.write(str(computeGain(prev)))
                fw.write("\n")
                print(f'[{model_name}] Current Sample {i}')



if __name__=="__main__":
    compute_doc_ref_similarity()
    compute_gt()