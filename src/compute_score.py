import jsonlines
import json
import pandas as pd
import ast

k = 3


# Read gain file for a given model
def read_gt_gain(model_name):
    """
    :rtype: List[List[tuple]]
    """
    dir = "./models/"
    # If summary_type is provided, use the category-aware file
    def _read(model_name, summary_type=None):
        if summary_type:
            fn = dir+f"{model_name}_gain_{summary_type.lower()}.txt"
        else:
            fn = dir+f"{model_name}_gain.txt"
        gt_gain = []
        with open(fn, "r") as file:
            for line in file:
                gt_gain.append(ast.literal_eval(line))
        return gt_gain
    return _read

def read_model_file(file="./models/model.json", summary_type=None):
    """
    :rtype: List[List[int]]
    """
    # If summary_type is provided, use the category-aware file
    if summary_type:
        file = f"./models/model_{summary_type.lower()}.json"
    data = json.load(open(file, "r"))
    df = pd.DataFrame.from_dict(data)
    return df["modelSenID"]


def scoreNCG(model_relevance, GT_relevance):
  """
  model_relevance: List[gain] -> for model
  GT_relevance: List[gain] -> ideal
  """

  # CG score 
  cg = sum(model_relevance)

  # ICG score 
  icg = sum(GT_relevance)

  # Normalized CG score 
  ncg = cg/icg

  return ncg

def computeNCG(gt, model):
    """
    #For a sample
    :type gt: List[tuple]
    :type model: List[int]
    :rtype: float -> ncg score
    """
    gt_dic = {int(k): v for k, v in dict(gt).items()}
    gt_rel = [v for _, v in gt[:k]]

    # model_rel = [gt_dic[id] for id in model]

    model_rel = []
    for j in range(k):
        model_rel.append(gt_dic[model[j]]) #take k sentences for model
    return scoreNCG(model_rel, gt_rel)


# Evaluate for all LLM_MODELS
def eval_ncg(summary_type=None):
    results = {}
    LLM_MODELS = [
        # LLMs
        # 'llama3.2',
        # 'llama2',
        # 'gemma-3-1b-it',
        # 'mistral',
        # 'openelm',
        # 'olmo-2-1b',
        # 'qwen3-0.6b',
        # Classical models
        # 'sbert-mini',
        # 'laser',
        # 'use',
        'roberta',
        # 'sbert-l',
        # 'simcse',
        # 'infersent'
    ]
    for model_name in LLM_MODELS:
        print(f"[INFO] Evaluating model: {model_name} | Category: {summary_type if summary_type else 'All'}")
        try:
            gt_gain = read_gt_gain(model_name)(model_name, summary_type=summary_type)
        except FileNotFoundError:
            print(f"[SKIP] Model {model_name}: gain file not found for {summary_type}")
            continue
        try:
            model_senId = read_model_file(summary_type=summary_type)
        except Exception as e:
            print(f"[SKIP] Model {model_name}: model file error: {e}")
            continue
        res = []
        for i in range(len(gt_gain)):
            if model_senId[i] is None:
                print(f"[SKIP] Sample {i}: model_senId is None")
                continue
            if len(model_senId[i]) < k:
                print(f"[SKIP] Sample {i}: model_senId has less than k elements")
                continue
            score = computeNCG(gt_gain[i], model_senId[i])
            if score is not None:
                res.append(score)
        # Always append _extractive or _abstractive to the label
        label = f"Sem-nCG@3_{model_name}_{summary_type.lower() if summary_type else 'all'}"
        results[label] = res
        # print(f"[RESULT] {model_name} ({summary_type if summary_type else 'All'}): {res}")
    return results

if __name__=='__main__':
    # Choose which categories to run
    categories = ["Extractive"]
    # categories = ["Abstractive"]
    # categories = ["Extractive","Abstractive"]
    with jsonlines.open("./output/score.jsonl", "a") as writer:
        for cat in categories:
            writer.write(eval_ncg(summary_type=cat))