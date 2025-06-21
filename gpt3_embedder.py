import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk import tokenize
from copy import deepcopy
from dotenv import load_dotenv
import openai
import torch
import re

def get_gpt3_embeddings(sentences, model="text-embedding-ada-002"):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=sentences,
        model=model
    )
    return [item['embedding'] for item in response['data']]

def mean_pooling(embeddings):
    # For GPT-3 embeddings, each sentence is already a vector
    return torch.tensor(embeddings)

def doc_per_sent_similarity(doc_sent_embeddings, summary_sent_embeddings):
    doc_sent_similarity = {}
    sen_id = 0
    for e1 in doc_sent_embeddings:
        e2_cos_sum = 0
        for e2 in summary_sent_embeddings:
            e1_cpu = e1.reshape(1, -1)
            e2_cpu = e2.reshape(1, -1)
            e2_cos_sum += cosine_similarity(e1_cpu, e2_cpu)[0][0]
        doc_sent_similarity[sen_id] = e2_cos_sum / summary_sent_embeddings.shape[0]
        sen_id += 1
    return doc_sent_similarity

def compute_similarity(doc_sent, ref_sent):
    doc_sent_embeddings = get_gpt3_embeddings(doc_sent)
    ref_sent_embeddings = get_gpt3_embeddings(ref_sent)
    doc_sent_embeddings = mean_pooling(doc_sent_embeddings)
    ref_sent_embeddings = mean_pooling(ref_sent_embeddings)
    return doc_per_sent_similarity(doc_sent_embeddings, ref_sent_embeddings)

def read_sample_file(file="sample.json"):
    data = json.load(open(file, "r"))
    df = pd.DataFrame.from_dict(data)
    return df["Doc"], df["Reference"], df["model"]

def compute_senID(doc_sent, model_summary_sent):
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
    if len(model_sum_SI) < len(model_summary_sent):
        print(f"[SKIP] Only matched {len(model_sum_SI)} of {len(model_summary_sent)}")
        return None
    return model_sum_SI

def compute_model_senId(doc, model):
    modelSenId = []
    for i in range(len(model)):
        doc_sent = tokenize.sent_tokenize(doc[i])
        model_summary_sent = tokenize.sent_tokenize(model[i])
        model_sum_SI = compute_senID(doc_sent, model_summary_sent)
        modelSenId.append(model_sum_SI)
    df = pd.DataFrame({'model': model, 'modelSenID': modelSenId})
    df.to_json("./output/model.json", orient='records')
    return

def compute_doc_ref_similarity():
    dir = "output"
    if not os.path.exists(dir):
        os.makedirs(dir)
    doc, ref, model = read_sample_file()
    compute_model_senId(doc, model)
    model_name = "gpt3"
    fn = f"./output/{model_name}_similarity.txt"
    with open(fn, "w") as fw:
        for i in range(len(doc)):
            doc_sent = tokenize.sent_tokenize(doc[i])
            ref_sent = tokenize.sent_tokenize(ref[i])
            sim = compute_similarity(doc_sent, ref_sent)
            sim = {k: float(v) for k, v in sim.items()}
            json.dump(sim, fw)
            fw.write("\n")

def read_file(model_name):
    dir = "./output/"
    fn = dir+f"{model_name}_similarity.txt"
    dicSim = []
    with open(fn, "r") as file:
        for line in file:
            dicSim.append(json.loads(line))
    fileId = [i for i in range(len(dicSim))]
    df = pd.DataFrame({"Idx" : fileId, "DictSim": dicSim})
    return df

def computeGain(dic_tuple):
    denom = 1.0*sum([i+1 for i in range(len(dic_tuple))])
    new_dic = {}
    g = len(dic_tuple)
    for sId, score in dic_tuple:
        new_dic[sId] = (g*1.0)/denom
        g-=1
    return sorted(new_dic.items(), key = lambda i : i[1], reverse=True)

def compute_gt():
    dir = "./output/"
    model_name = "gpt3"
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
    load_dotenv()
    compute_doc_ref_similarity()
    compute_gt()
