import jsonlines
import json
import pandas as pd
import ast
import re

k = 3


# Read gain file for a given model
def read_gt_gain(model_name):
    """
    :rtype: List[List[tuple]]
    """
    dir = "./output/"
    fn = dir+f"{model_name}_gain.txt"
    gt_gain = []
    with open(fn, "r") as file:
        for line in file:
            gt_gain.append(ast.literal_eval(line))
    return gt_gain

def read_model_file(file="./output/model.json"):
    """
    :rtype: List[List[int]]
    """
    data = json.load(open(file, "r"))
    df = pd.DataFrame.from_dict(data) #orient="index" must be provided if the json file is in indexed format
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



# Evaluate for all LLM_MODELS from pre-run.py
def get_llm_model_names():
    # Read LLM_MODELS from pre-run.py and extract the model short names
    with open('pre-run.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    model_names = []
    in_models = False
    for line in lines:
        if 'LLM_MODELS' in line and '=' in line:
            in_models = True
            continue
        if in_models:
            if ']' in line:
                break
            # Match lines like: ('model-path', 'shortname'),
            match = re.search(r"'[^']+'\s*,\s*'([^']+)'", line)
            if match:
                model_names.append(match.group(1))
    return model_names

def eval_ncg():
    results = {}
    LLM_MODELS = get_llm_model_names()
    for model_name in LLM_MODELS:
        gt_gain = read_gt_gain(model_name)
        model_senId = read_model_file()  # model.json is shared
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
        results[f"Sem-nCG@3_{model_name}"] = res
    return results


if __name__=='__main__':
    with jsonlines.open("score.jsonl", "a") as writer:   # for writing
        writer.write(eval_ncg())